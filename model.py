import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import WavLMModel
from mamba_blocks import MixerModel


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        pred = pred.squeeze(-1)
        target = target.squeeze(-1).float()
        p = torch.sigmoid(pred)
        ce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        p_t = p * target + (1.0 - p) * (1.0 - target)
        alpha_t = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)
        loss = alpha_t * (1.0 - p_t).pow(self.gamma) * ce
        return loss.sum() if self.reduction == 'sum' else loss.mean()


class P2SGradLoss(nn.Module):
    def __init__(self, scale=30.0):
        super().__init__()
        self.scale = scale
        self.ce = nn.CrossEntropyLoss()

    def forward(self, feat, target, weight_matrix):
        feat_norm = F.normalize(feat, dim=-1)
        weight_norm = F.normalize(weight_matrix, dim=-1)
        scores = torch.matmul(feat_norm, weight_norm.t()) * self.scale
        bsz, num_frames, num_classes = scores.shape
        return self.ce(scores.reshape(bsz * num_frames, num_classes), target.reshape(bsz * num_frames))


class FrozenWavLMFrontend(nn.Module):
    def __init__(self, model_name='microsoft/wavlm-large', target_frames=208):
        super().__init__()
        self.model = WavLMModel.from_pretrained(model_name)
        self.model.eval()
        self.target_frames = target_frames
        self.out_dim = self.model.config.hidden_size
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, waveform):
        if waveform.dim() == 3:
            waveform = waveform.squeeze(-1)
        with torch.no_grad():
            features = self.model(input_values=waveform).last_hidden_state
        if features.size(1) != self.target_frames:
            features = F.interpolate(features.transpose(1, 2), size=self.target_frames, mode='linear', align_corners=False).transpose(1, 2)
        return features


class LogMelCNNFrontend(nn.Module):
    def __init__(self, d_model, sample_rate=16000, n_fft=400, hop_length=320, win_length=400, n_mels=128, target_frames=208):
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, win_length=win_length, hop_length=hop_length,
            f_min=20.0, f_max=7600.0, n_mels=n_mels, power=2.0, center=False, normalized=False,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80.0)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, 3), padding=(2, 1), bias=False),
            nn.BatchNorm2d(32), nn.SiLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=(5, 3), padding=(2, 1), bias=False),
            nn.BatchNorm2d(64), nn.SiLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64), nn.SiLU(inplace=True),
        )
        self.freq_pool = nn.AdaptiveAvgPool2d((1, target_frames))
        self.proj = nn.Linear(64, d_model)

    def forward(self, waveform):
        if waveform.dim() == 3:
            waveform = waveform.squeeze(-1)
        x = self.to_db(self.mel_spec(waveform).clamp_min(1e-5)).unsqueeze(1)
        x = self.cnn(x)
        x = self.freq_pool(x).squeeze(2).transpose(1, 2)
        return self.proj(x)


class BoundaryAwareHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        hidden = max(d_model // 2, 32)
        self.net = nn.Sequential(
            nn.Conv1d(d_model, hidden, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(hidden), nn.SiLU(inplace=True),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(hidden), nn.SiLU(inplace=True),
            nn.Conv1d(hidden, 1, kernel_size=1),
        )

    def forward(self, h):
        return self.net(h.transpose(1, 2)).transpose(1, 2)


class BoundaryControlledMixer(nn.Module):
    def __init__(self, d_model, n_layer, enabled=True):
        super().__init__()
        self.enabled = enabled
        self.mixer = MixerModel(d_model=d_model, n_layer=n_layer, ssm_cfg={}, rms_norm=False, residual_in_fp32=True, fused_add_norm=False)
        self.control = nn.Sequential(nn.Linear(d_model + 1, d_model), nn.SiLU(inplace=True), nn.Linear(d_model, d_model), nn.Sigmoid())
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, boundary_prob):
        mixed = self.mixer(x)
        if self.enabled:
            gate = self.control(torch.cat([x, boundary_prob], dim=-1))
            out = x + gate * (mixed - x)
        else:
            gate = torch.ones_like(x)
            out = mixed
        return self.norm(out), gate


class BoundaryRoutedCrossStream(nn.Module):
    def __init__(self, d_model, enabled=True):
        super().__init__()
        self.enabled = enabled
        self.time_to_freq = nn.Linear(d_model, d_model)
        self.freq_to_time = nn.Linear(d_model, d_model)
        self.route = nn.Sequential(nn.Linear(d_model * 2 + 1, d_model), nn.SiLU(inplace=True), nn.Linear(d_model, d_model), nn.Sigmoid())
        self.time_norm = nn.LayerNorm(d_model)
        self.freq_norm = nn.LayerNorm(d_model)

    def forward(self, h_time, h_freq, boundary_prob):
        if self.enabled:
            route = self.route(torch.cat([h_time, h_freq, boundary_prob], dim=-1))
            new_time = self.time_norm(h_time + route * self.freq_to_time(h_freq))
            new_freq = self.freq_norm(h_freq + route * self.time_to_freq(new_time))
        else:
            route = torch.zeros_like(h_time)
            new_time = self.time_norm(h_time)
            new_freq = self.freq_norm(h_freq)
        return new_time, new_freq, route


class SoftSegmentParser(nn.Module):
    def __init__(self, d_model, num_segments=4, enabled=True):
        super().__init__()
        self.enabled = enabled
        self.num_segments = num_segments
        self.assign = nn.Linear(d_model + 1, num_segments)
        self.refine = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.SiLU(inplace=True), nn.Linear(d_model, d_model))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h_frame, boundary_prob):
        if self.enabled:
            assign_logits = self.assign(torch.cat([h_frame, boundary_prob], dim=-1))
            assign = F.softmax(assign_logits, dim=-1)
            denom = assign.sum(dim=1, keepdim=True).transpose(1, 2).clamp_min(1e-5)
            segments = torch.matmul(assign.transpose(1, 2), h_frame) / denom
            segment_back = torch.matmul(assign, segments)
            h_segment = self.norm(h_frame + self.refine(torch.cat([h_frame, segment_back], dim=-1)))
            seg_consistency = ((h_frame - segment_back) ** 2).mean()
            seg_entropy = -(assign * (assign.clamp_min(1e-8).log())).sum(dim=-1).mean()
        else:
            batch_size, num_frames, d_model = h_frame.shape
            assign = h_frame.new_zeros(batch_size, num_frames, self.num_segments)
            assign[..., 0] = 1.0
            segments = h_frame.mean(dim=1, keepdim=True).expand(-1, self.num_segments, -1)
            segment_back = h_frame
            h_segment = self.norm(h_frame)
            seg_consistency = h_frame.new_zeros(())
            seg_entropy = h_frame.new_zeros(())
        return h_segment, {'segment_consistency': seg_consistency, 'segment_entropy': seg_entropy, 'assignments': assign, 'segments': segments, 'segment_back': segment_back}


class HierarchicalAttractorHead(nn.Module):
    def __init__(self, d_model, num_classes=3, num_heads=4, dropout=0.1, num_segments=4, use_soft_segments=True):
        super().__init__()
        self.num_classes = num_classes
        self.attractor_tokens = nn.Parameter(torch.randn(num_classes, d_model) * 0.02)
        self.segment_tokens = nn.Parameter(torch.randn(num_classes, d_model) * 0.02)
        self.frame_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.segment_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.segment_parser = SoftSegmentParser(d_model=d_model, num_segments=num_segments, enabled=use_soft_segments)
        self.frame_norm = nn.LayerNorm(d_model)
        self.segment_norm = nn.LayerNorm(d_model)
        self.fusion = nn.Sequential(nn.Linear(d_model * 3, d_model), nn.SiLU(inplace=True), nn.Linear(d_model, d_model))
        self.fusion_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, h_enhanced, boundary_prob):
        batch_size = h_enhanced.size(0)
        frame_tokens = self.attractor_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        frame_ctx, frame_attn_map = self.frame_attn(query=h_enhanced, key=frame_tokens, value=frame_tokens, need_weights=True, average_attn_weights=False)
        h_frame = self.frame_norm(h_enhanced + frame_ctx)
        stable_weight = (1.0 - boundary_prob).clamp_min(1e-4)
        stable_summary = (h_frame * stable_weight).sum(dim=1, keepdim=True) / stable_weight.sum(dim=1, keepdim=True)
        segment_tokens = self.segment_tokens.unsqueeze(0).expand(batch_size, -1, -1) + stable_summary
        segment_ctx, segment_attn_map = self.segment_attn(query=h_frame, key=segment_tokens, value=segment_tokens, need_weights=True, average_attn_weights=False)
        h_context = self.segment_norm(h_frame + segment_ctx)
        h_segment, aux = self.segment_parser(h_context, boundary_prob)
        h_prime = self.fusion_norm(h_segment + self.fusion(torch.cat([h_frame, h_context, h_segment], dim=-1)))
        aux['frame_attn_map'] = frame_attn_map
        aux['segment_attn_map'] = segment_attn_map
        aux['stable_summary'] = stable_summary
        return self.classifier(h_prime), h_prime, aux


class Model(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.device = device
        self.d_model = getattr(args, 'emb_size', 144)
        self.num_classes = getattr(args, 'num_classes', 3)
        self.num_frames = 208
        self.cached_aux_losses = {}
        self.cached_analysis = {}
        self.use_boundary_control = getattr(args, 'use_boundary_control', True)
        self.use_cross_routing = getattr(args, 'use_cross_routing', True)
        self.use_soft_segments = getattr(args, 'use_soft_segments', True)
        num_layers = max(getattr(args, 'num_encoders', 12) // 2, 1)
        num_heads = self._pick_num_heads(self.d_model)
        num_segments = getattr(args, 'num_segments', 4)

        self.ssl_model = FrozenWavLMFrontend(model_name='microsoft/wavlm-large', target_frames=self.num_frames)
        self.time_proj = nn.Linear(self.ssl_model.out_dim, self.d_model)
        self.time_norm = nn.LayerNorm(self.d_model)
        self.freq_stream = LogMelCNNFrontend(d_model=self.d_model, sample_rate=16000, n_fft=400, hop_length=320, win_length=400, n_mels=128, target_frames=self.num_frames)
        self.freq_norm = nn.LayerNorm(self.d_model)
        self.prior_fusion = nn.Sequential(nn.Linear(self.d_model * 2, self.d_model), nn.SiLU(inplace=True), nn.LayerNorm(self.d_model))
        self.boundary_prior_head = BoundaryAwareHead(self.d_model)
        self.time_mamba = BoundaryControlledMixer(self.d_model, num_layers, enabled=self.use_boundary_control)
        self.freq_mamba = BoundaryControlledMixer(self.d_model, num_layers, enabled=self.use_boundary_control)
        self.cross_stream = BoundaryRoutedCrossStream(self.d_model, enabled=self.use_cross_routing)
        self.fusion = nn.Sequential(nn.Linear(self.d_model * 2, self.d_model), nn.SiLU(inplace=True), nn.Linear(self.d_model, self.d_model))
        self.fusion_norm = nn.LayerNorm(self.d_model)
        self.boundary_head = BoundaryAwareHead(self.d_model)
        self.attractor_head = HierarchicalAttractorHead(d_model=self.d_model, num_classes=self.num_classes, num_heads=num_heads, dropout=0.1, num_segments=num_segments, use_soft_segments=self.use_soft_segments)

        print('BATS-Mamba: Boundary-Controlled State Parsing for Spoof Diarization')
        print(f'd_model={self.d_model}, num_layers={num_layers}, num_heads={num_heads}, num_segments={num_segments}, num_classes={self.num_classes}')

    @staticmethod
    def _pick_num_heads(d_model):
        for candidate in [8, 6, 4, 3, 2, 1]:
            if d_model % candidate == 0:
                return candidate
        return 1

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(-1)
        time_feat = self.time_norm(self.time_proj(self.ssl_model(x)))
        freq_feat = self.freq_norm(self.freq_stream(x))
        prior_feat = self.prior_fusion(torch.cat([time_feat, freq_feat], dim=-1))
        p_bound_prior = torch.sigmoid(self.boundary_prior_head(prior_feat))
        h_time, time_gate = self.time_mamba(time_feat, p_bound_prior)
        h_freq, freq_gate = self.freq_mamba(freq_feat, p_bound_prior)
        h_time, h_freq, route_map = self.cross_stream(h_time, h_freq, p_bound_prior)
        h_fusion = self.fusion_norm(self.fusion(torch.cat([h_time, h_freq], dim=-1)))
        p_bound_logits = self.boundary_head(h_fusion)
        p_bound = torch.sigmoid(p_bound_logits)
        h_enhanced = h_fusion + (h_fusion * p_bound)
        logits_dia, h_prime, aux_losses = self.attractor_head(h_enhanced, p_bound)
        boundary_shift = (p_bound[:, 1:] - p_bound[:, :-1]).abs().mean()
        self.cached_aux_losses = {
            'segment_consistency': aux_losses['segment_consistency'],
            'segment_entropy': aux_losses['segment_entropy'],
            'boundary_sparsity': p_bound.mean(),
            'boundary_sharpness': -boundary_shift,
        }
        self.cached_analysis = {
            'boundary_prior': p_bound_prior.detach(),
            'boundary_final': p_bound.detach(),
            'time_gate': time_gate.detach(),
            'freq_gate': freq_gate.detach(),
            'route_map': route_map.detach(),
            'segment_assignments': aux_losses['assignments'].detach(),
            'segment_centers': aux_losses['segments'].detach(),
            'segment_backproj': aux_losses['segment_back'].detach(),
            'frame_attn_map': aux_losses['frame_attn_map'].detach(),
            'segment_attn_map': aux_losses['segment_attn_map'].detach(),
            'stable_summary': aux_losses['stable_summary'].detach(),
        }
        return p_bound_logits, logits_dia, h_prime

    def compute_spoof_ratio(self, logits_dia):
        preds = logits_dia.argmax(dim=-1)
        batch_ratios = []
        for sample_pred in preds:
            class_ratios = {}
            for cls_idx in range(self.num_classes):
                class_ratios[cls_idx] = (sample_pred == cls_idx).float().mean().item() * 100.0
            batch_ratios.append(class_ratios)
        return batch_ratios
