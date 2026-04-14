# BAT-Mamba: main.py
# Multi-task progressive training:
#   Phase 1 (ep 1-5):   L = L_loc                        lambda=(1,0,0)
#   Phase 2 (ep 6-15):  L = L_loc + L_bound              lambda=(1,1,0)
#   Phase 3 (ep 16+):   L = L_loc + L_bound + L_dia      lambda=(1,1,1)

import argparse
import sys
import os
import json
import shutil
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_utils import (
    Dataset_train, Dataset_eval, Dataset_in_the_wild_eval, genSpoof_list,
    Dataset_PartialSpoof_train, Dataset_PartialSpoof_eval,
    load_seglab, parse_ps_protocol, NUM_FRAMES
)
from model import Model, FocalLoss, P2SGradLoss
from utils import reproducibility, read_metadata
import numpy as np


def compute_multiclass_f1(y_true, y_pred, num_classes):
    f1_scores = []
    for cls_idx in range(num_classes):
        tp = np.sum((y_true == cls_idx) & (y_pred == cls_idx))
        fp = np.sum((y_true != cls_idx) & (y_pred == cls_idx))
        fn = np.sum((y_true == cls_idx) & (y_pred != cls_idx))
        denom = 2 * tp + fp + fn
        f1_scores.append(0.0 if denom == 0 else (2.0 * tp) / denom)
    return float(np.mean(f1_scores)), f1_scores


def compute_binary_prf(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    precision = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
    recall = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
    f1 = 0.0 if (precision + recall) == 0 else 2.0 * precision * recall / (precision + recall)
    return float(precision), float(recall), float(f1)


def summarize_eval_metrics(frame_true, frame_pred, bound_true, bound_pred, num_classes):
    frame_true = np.asarray(frame_true, dtype=np.int64)
    frame_pred = np.asarray(frame_pred, dtype=np.int64)
    bound_true = np.asarray(bound_true, dtype=np.int64)
    bound_pred = np.asarray(bound_pred, dtype=np.int64)

    frame_acc = float((frame_true == frame_pred).mean()) if frame_true.size > 0 else 0.0
    macro_f1, class_f1 = compute_multiclass_f1(frame_true, frame_pred, num_classes=num_classes)
    spoof_precision, spoof_recall, spoof_f1 = compute_binary_prf((frame_true == 1).astype(np.int64), (frame_pred == 1).astype(np.int64))
    bound_precision, bound_recall, bound_f1 = compute_binary_prf(bound_true, bound_pred)

    return {
        'frame_acc': frame_acc,
        'macro_f1': macro_f1,
        'spoof_precision': spoof_precision,
        'spoof_recall': spoof_recall,
        'spoof_f1': spoof_f1,
        'boundary_precision': bound_precision,
        'boundary_recall': bound_recall,
        'boundary_f1': bound_f1,
        'class_f1': class_f1,
    }


def get_experiment_config(args, model_tag):
    return {
        'model_tag': model_tag,
        'comment': args.comment,
        'seed': args.seed,
        'emb_size': args.emb_size,
        'num_encoders': args.num_encoders,
        'num_classes': args.num_classes,
        'num_segments': args.num_segments,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'batch_size': args.batch_size,
        'use_boundary_control': args.use_boundary_control,
        'use_cross_routing': args.use_cross_routing,
        'use_soft_segments': args.use_soft_segments,
        'seg_cons_weight': args.seg_cons_weight,
        'seg_entropy_weight': args.seg_entropy_weight,
        'bound_sparse_weight': args.bound_sparse_weight,
        'bound_sharp_weight': args.bound_sharp_weight,
        'debug_steps': args.debug_steps,
    }


def append_experiment_record(record_path, payload):
    if os.path.exists(record_path):
        with open(record_path, 'r', encoding='utf-8') as fh:
            records = json.load(fh)
    else:
        records = []
    records.append(payload)
    with open(record_path, 'w', encoding='utf-8') as fh:
        json.dump(records, fh, indent=2)


def _to_serializable_config(config):
    serializable = {}
    for key, value in config.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            serializable[key] = value
        elif isinstance(value, (list, tuple)):
            serializable[key] = list(value)
        else:
            serializable[key] = str(value)
    return serializable


def export_analysis_artifacts(dataset, model, device, export_dir, max_batches=1, batch_size=4):
    os.makedirs(export_dir, exist_ok=True)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    model.eval()
    exported = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if batch_idx >= max_batches:
                break
            if len(batch) == 4:
                batch_x, batch_fl, batch_bl, utt_id = batch
            else:
                continue
            batch_x = batch_x.to(device)
            _ = model(batch_x)
            analysis = getattr(model, 'cached_analysis', {})
            if not analysis:
                continue
            batch_dir = os.path.join(export_dir, 'batch_{:03d}'.format(batch_idx))
            os.makedirs(batch_dir, exist_ok=True)
            meta = {
                'batch_idx': int(batch_idx),
                'utt_id': list(utt_id),
                'frame_labels_shape': list(batch_fl.shape),
                'boundary_labels_shape': list(batch_bl.shape),
                'analysis_keys': sorted(list(analysis.keys())),
            }
            with open(os.path.join(batch_dir, 'meta.json'), 'w', encoding='utf-8') as fh:
                json.dump(meta, fh, indent=2)
            np.save(os.path.join(batch_dir, 'frame_labels.npy'), batch_fl.numpy())
            np.save(os.path.join(batch_dir, 'boundary_labels.npy'), batch_bl.numpy())
            for key, tensor in analysis.items():
                np.save(os.path.join(batch_dir, '{}.npy'.format(key)), tensor.detach().cpu().numpy())
            exported += 1
    print('Exported analysis batches: {} -> {}'.format(exported, export_dir))


# ============================================================
# Helper: get progressive loss weights by epoch
# ============================================================
def get_loss_weights(epoch):
    """BAT-Mamba V2 / Phase 1 recommended progressive schedule."""
    if epoch < 5:
        return 1.0, 0.0, 0.0   # Stage 1: localization only
    elif epoch < 15:
        return 1.0, 1.0, 0.5   # Stage 2: activate boundary + mild metric learning
    else:
        return 1.0, 1.0, 1.0   # Stage 3: full supervision


def get_loss_weights_debug(epoch):
    """Debug mode follows the same schedule as formal training."""
    return get_loss_weights(epoch)


# make_frame_labels removed: PartialSpoof Dataset now returns real frame labels directly.


# ============================================================
# Evaluation helpers
# ============================================================
def evaluate_accuracy(dev_loader, model, device, debug_steps=0):
    """Validation loop. dev_loader yields (waveform, frame_labels, boundary_labels, utt_id).
    debug_steps: if > 0, only run this many batches.
    """
    val_loss = 0.0
    num_total = 0.0
    all_frame_true, all_frame_pred = [], []
    all_bound_true, all_bound_pred = [], []
    model.eval()
    criterion_loc = nn.CrossEntropyLoss()
    num_batch = len(dev_loader)
    with torch.no_grad():
        for i, batch_data in enumerate(dev_loader):
            if debug_steps > 0 and i >= debug_steps:
                break
            if len(batch_data) == 4:
                batch_x, batch_fl, batch_bl, _ = batch_data
            else:
                batch_x, batch_fl, batch_bl = batch_data

            batch_size = batch_x.size(0)
            num_total += batch_size
            batch_x = batch_x.to(device)
            batch_fl = batch_fl.to(device)
            batch_bl = batch_bl.to(device)

            p_bound_logits, logits_dia, _ = model(batch_x)
            B, T, C = logits_dia.shape
            loss = criterion_loc(
                logits_dia.reshape(B * T, C),
                batch_fl.reshape(B * T)
            )
            val_loss += loss.item() * batch_size

            frame_pred = logits_dia.argmax(dim=-1)
            bound_pred = (torch.sigmoid(p_bound_logits).squeeze(-1) >= 0.5).long()
            all_frame_true.append(batch_fl.detach().cpu().reshape(-1).numpy())
            all_frame_pred.append(frame_pred.detach().cpu().reshape(-1).numpy())
            all_bound_true.append(batch_bl.detach().cpu().reshape(-1).numpy())
            all_bound_pred.append(bound_pred.detach().cpu().reshape(-1).numpy())
            print("batch %i/%i (val)" % (i + 1, num_batch), end="\r")

    val_loss /= num_total
    metrics = summarize_eval_metrics(
        np.concatenate(all_frame_true) if all_frame_true else np.array([], dtype=np.int64),
        np.concatenate(all_frame_pred) if all_frame_pred else np.array([], dtype=np.int64),
        np.concatenate(all_bound_true) if all_bound_true else np.array([], dtype=np.int64),
        np.concatenate(all_bound_pred) if all_bound_pred else np.array([], dtype=np.int64),
        num_classes=model.num_classes,
    )
    print('Val loss: %.4f | FrameAcc: %.4f | MacroF1: %.4f | SpoofF1: %.4f | BoundF1: %.4f' % (
        val_loss, metrics['frame_acc'], metrics['macro_f1'], metrics['spoof_f1'], metrics['boundary_f1']))
    print('          SpoofP/R: %.4f / %.4f | BoundP/R: %.4f / %.4f | ClassF1=%s' % (
        metrics['spoof_precision'], metrics['spoof_recall'],
        metrics['boundary_precision'], metrics['boundary_recall'],
        ','.join(['%.4f' % x for x in metrics['class_f1']])
    ))
    return val_loss, metrics


def produce_evaluation_file(dataset, model, device, save_path):
    """Evaluation file writer. dataset yields (waveform, frame_labels, boundary_labels, utt_id)."""
    data_loader = DataLoader(dataset, batch_size=40, shuffle=False, drop_last=False)
    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader, total=len(data_loader)):
            # Support both 4-tuple (PartialSpoof eval) and 2-tuple (legacy eval)
            if len(batch) == 4:
                batch_x, _, _, utt_id = batch
            else:
                batch_x, utt_id = batch
            batch_x = batch_x.to(device)
            _, logits_dia, _ = model(batch_x)
            # Sentence-level score: average frame logit[:,1] (spoof class)
            batch_score = logits_dia[:, :, 1].mean(dim=1).data.cpu().numpy().ravel()
            with open(save_path, 'a+') as fh:
                for f, cm in zip(utt_id, batch_score.tolist()):
                    fh.write('{} {}\n'.format(f, cm))
    print('Scores saved to {}'.format(save_path))


# ============================================================
# Training epoch with progressive multi-task loss
# ============================================================
def train_epoch(train_loader, model, optimizer, device, epoch, checkpoint_dir=None, debug_steps=0, args=None):
    """PartialSpoof training epoch.
    train_loader yields: (waveform [B,66800], frame_labels [B,208], boundary_labels [B,208])
    checkpoint_dir: if set, saves model every 1000 steps to prevent data loss.
    debug_steps: if > 0, only run this many batches (quick pipeline test).
    """
    model.train()
    num_total = 0.0
    total_loss = 0.0
    criterion_loc   = nn.CrossEntropyLoss()
    criterion_bound = FocalLoss(alpha=0.25, gamma=2.0)
    criterion_dia   = P2SGradLoss(scale=30.0)
    seg_cons_weight = getattr(args, 'seg_cons_weight', 0.0) if args is not None else 0.0
    seg_entropy_weight = getattr(args, 'seg_entropy_weight', 0.0) if args is not None else 0.0
    bound_sparse_weight = getattr(args, 'bound_sparse_weight', 0.0) if args is not None else 0.0
    bound_sharp_weight = getattr(args, 'bound_sharp_weight', 0.0) if args is not None else 0.0
    lam1, lam2, lam3 = get_loss_weights_debug(epoch) if debug_steps > 0 else get_loss_weights(epoch)
    print('Phase weights: lam1=%.1f  lam2=%.1f  lam3=%.1f' % (lam1, lam2, lam3))
    scaler = torch.cuda.amp.GradScaler()  # AMP scaler
    pbar = tqdm(train_loader, total=len(train_loader))
    for step, (batch_x, batch_fl, batch_bl) in enumerate(pbar):
        if debug_steps > 0 and step >= debug_steps:
            break
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x  = batch_x.to(device)
        batch_fl = batch_fl.to(device)
        batch_bl = batch_bl.unsqueeze(-1).to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():  # AMP: FP16 forward pass
            p_bound_logits, logits_dia, h_prime = model(batch_x)
            B, T, C = logits_dia.shape

            loss_loc = criterion_loc(
                logits_dia.reshape(B * T, C),
                batch_fl.reshape(B * T)
            )
            loss_bound = criterion_bound(p_bound_logits, batch_bl) if lam2 > 0 \
                else torch.tensor(0.0, device=device)
            loss_dia = criterion_dia(
                h_prime, batch_fl,
                model.attractor_head.attractor_tokens
            ) if lam3 > 0 else torch.tensor(0.0, device=device)

            loss = lam1 * loss_loc + lam2 * loss_bound + lam3 * loss_dia

            aux_losses = getattr(model, 'cached_aux_losses', {})
            loss_seg_cons = aux_losses.get('segment_consistency', torch.tensor(0.0, device=device))
            loss_seg_entropy = aux_losses.get('segment_entropy', torch.tensor(0.0, device=device))
            loss_bound_sparse = aux_losses.get('boundary_sparsity', torch.tensor(0.0, device=device))
            loss_bound_sharp = aux_losses.get('boundary_sharpness', torch.tensor(0.0, device=device))

            loss = loss \
                + seg_cons_weight * loss_seg_cons \
                + seg_entropy_weight * loss_seg_entropy \
                + bound_sparse_weight * loss_bound_sparse \
                + bound_sharp_weight * loss_bound_sharp

        total_loss += loss.item() * batch_size
        scaler.scale(loss).backward()   # AMP: scaled backward
        scaler.step(optimizer)          # AMP: scaled optimizer step
        scaler.update()                 # AMP: update scaler

        pbar.set_postfix({
            'loss': '%.4f' % (total_loss / num_total),
            'loc':  '%.4f' % loss_loc.item(),
            'bnd':  '%.4f' % loss_bound.item(),
            'dia':  '%.4f' % loss_dia.item(),
        })

        # Save checkpoint every 1000 steps, keep only the latest one
        if checkpoint_dir is not None and (step + 1) % 1000 == 0:
            ckpt_path = os.path.join(checkpoint_dir,
                'checkpoint_ep{}_step{}.pth'.format(epoch, step + 1))
            torch.save(model.state_dict(), ckpt_path)
            print('\nCheckpoint saved: {}'.format(ckpt_path))
            # Delete previous checkpoint for this epoch to save disk space
            prev_step = step + 1 - 1000
            if prev_step > 0:
                prev_ckpt = os.path.join(checkpoint_dir,
                    'checkpoint_ep{}_step{}.pth'.format(epoch, prev_step))
                if os.path.exists(prev_ckpt):
                    os.remove(prev_ckpt)

    sys.stdout.flush()


# ============================================================
# Main entry
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BAT-Mamba')
    parser.add_argument('--database_path', type=str, default='./data/')
    parser.add_argument('--protocols_path', type=str, default='./data/')
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--num_epochs', type=int, default=25)
    parser.add_argument('--lr', type=float, default=0.000001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--loss', type=str, default='WCE')
    parser.add_argument('--emb-size', type=int, default=144)
    parser.add_argument('--num_encoders', type=int, default=12)
    parser.add_argument('--num_classes', type=int, default=3,
                        help='number of frame-level classes')
    parser.add_argument('--FT_W2V', default=True,
                        type=lambda x: (str(x).lower() in ['true','yes','1']))
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--comment', type=str, default=None)
    parser.add_argument('--debug_steps', type=int, default=0,
                        help='If > 0, only run this many batches per epoch/eval (quick pipeline test)')
    parser.add_argument('--comment_eval', type=str, default=None)
    parser.add_argument('--train', default=True,
                        type=lambda x: (str(x).lower() in ['true','yes','1']))
    parser.add_argument('--n_mejores_loss', type=int, default=5)
    parser.add_argument('--average_model', default=True,
                        type=lambda x: (str(x).lower() in ['true','yes','1']))
    parser.add_argument('--n_average_model', default=5, type=int)
    parser.add_argument('--algo', type=int, default=5)
    parser.add_argument('--N_f', type=int, default=5)
    parser.add_argument('--nBands', type=int, default=5)
    parser.add_argument('--minF', type=int, default=20)
    parser.add_argument('--maxF', type=int, default=8000)
    parser.add_argument('--minBW', type=int, default=100)
    parser.add_argument('--maxBW', type=int, default=1000)
    parser.add_argument('--minCoeff', type=int, default=10)
    parser.add_argument('--maxCoeff', type=int, default=100)
    parser.add_argument('--minG', type=int, default=0)
    parser.add_argument('--maxG', type=int, default=0)
    parser.add_argument('--minBiasLinNonLin', type=int, default=5)
    parser.add_argument('--maxBiasLinNonLin', type=int, default=20)
    parser.add_argument('--P', type=int, default=10)
    parser.add_argument('--g_sd', type=int, default=2)
    parser.add_argument('--SNRmin', type=int, default=10)
    parser.add_argument('--SNRmax', type=int, default=40)
    parser.add_argument('--num_segments', type=int, default=4,
                        help='number of latent soft segments for segment parsing')
    parser.add_argument('--seg_cons_weight', type=float, default=0.0,
                        help='weight for segment consistency loss')
    parser.add_argument('--seg_entropy_weight', type=float, default=0.0,
                        help='weight for soft segment assignment entropy')
    parser.add_argument('--bound_sparse_weight', type=float, default=0.0,
                        help='weight for boundary sparsity regularization')
    parser.add_argument('--bound_sharp_weight', type=float, default=0.0,
                        help='weight for boundary sharpness regularization')
    parser.add_argument('--use_boundary_control', default=True,
                        type=lambda x: (str(x).lower() in ['true','yes','1']))
    parser.add_argument('--use_cross_routing', default=True,
                        type=lambda x: (str(x).lower() in ['true','yes','1']))
    parser.add_argument('--use_soft_segments', default=True,
                        type=lambda x: (str(x).lower() in ['true','yes','1']))
    parser.add_argument('--export_analysis', default=False,
                        type=lambda x: (str(x).lower() in ['true','yes','1']))
    parser.add_argument('--analysis_max_batches', type=int, default=1,
                        help='number of evaluation batches to export analysis artifacts for')
    parser.add_argument('--analysis_batch_size', type=int, default=4,
                        help='batch size for analysis export loader')

    if not os.path.exists('models'):
        os.mkdir('models')
    args = parser.parse_args()
    args.track = 'LA'
    print(args)
    reproducibility(args.seed, args)

    track = args.track
    n_mejores = args.n_mejores_loss
    assert track in ['LA','DF','In-the-Wild'], 'Invalid track'
    assert args.n_average_model < args.n_mejores_loss + 1

    prefix_2019 = 'ASVspoof2019.{}'.format(track)
    model_tag = 'BATmamba{}_{}_{}_{}_ES{}_NE{}'.format(
        args.algo, track, args.loss, args.lr, args.emb_size, args.num_encoders)
    if args.comment:
        model_tag = model_tag + '_{}'.format(args.comment)
    model_save_path = os.path.join('models', model_tag)
    print('Model tag: ' + model_tag)
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    best_save_path = os.path.join(model_save_path, 'best')
    if not os.path.exists(best_save_path):
        os.mkdir(best_save_path)
    experiment_config = get_experiment_config(args, model_tag)
    experiment_config_path = os.path.join(model_save_path, 'experiment_config.json')
    with open(experiment_config_path, 'w', encoding='utf-8') as fh:
        json.dump(_to_serializable_config(experiment_config), fh, indent=2)
    experiment_record_path = os.path.join(model_save_path, 'validation_records.json')
    analysis_export_dir = os.path.join(model_save_path, 'analysis_exports')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: {}'.format(device))
    model = Model(args, device)
    if not args.FT_W2V:
        for param in model.ssl_model.parameters():
            param.requires_grad = False
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ---- In-the-Wild eval only ----
    if args.track == 'In-the-Wild':
        best_save_path = best_save_path.replace(track, 'LA')
        model_save_path = model_save_path.replace(track, 'LA')
        print('######## Eval In-the-Wild ########')
        model.load_state_dict(torch.load(
            os.path.join(best_save_path, 'best_0.pth')))
        sd = model.state_dict()
        for i in range(1, args.n_average_model):
            model.load_state_dict(torch.load(
                os.path.join(best_save_path, 'best_{}.pth'.format(i))))
            sd2 = model.state_dict()
            for key in sd:
                sd[key] = sd[key] + sd2[key]
        for key in sd:
            sd[key] = sd[key] / args.n_average_model
        model.load_state_dict(sd)
        file_eval = genSpoof_list(
            dir_meta=os.path.join(args.protocols_path),
            is_train=False, is_eval=True)
        eval_set = Dataset_in_the_wild_eval(
            list_IDs=file_eval, base_dir=os.path.join(args.database_path))
        produce_evaluation_file(eval_set, model, device,
            'Scores/{}/{}.txt'.format(args.track, model_tag))
        sys.exit(0)

    # ---- PartialSpoof Data loaders ----
    # Paths (based on H:\PS_data layout):
    #   audio:    database_path/{train,dev,eval}/con_wav/*.wav
    #   seglab:   database_path/segment_labels/{split}_seglab_0.02.npy
    #   protocol: protocols_path/PartialSpoof_LA_cm_protocols/PartialSpoof.LA.cm.{split}.trl.txt

    # Load segment labels (.npy) -- '0'=spoof, '1'=bonafide (will be flipped in Dataset)
    seglab_train = load_seglab(
        os.path.join(args.database_path, 'segment_labels', 'train_seglab_0.02.npy'))
    seglab_dev   = load_seglab(
        os.path.join(args.database_path, 'segment_labels', 'dev_seglab_0.02.npy'))

    # Parse CM protocol for audio IDs
    ps_proto_dir = os.path.join(args.protocols_path,
                                'protocols', 'PartialSpoof_LA_cm_protocols')
    files_id_train, _ = parse_ps_protocol(
        os.path.join(ps_proto_dir, 'PartialSpoof.LA.cm.train.trl.txt'))
    files_id_dev, _   = parse_ps_protocol(
        os.path.join(ps_proto_dir, 'PartialSpoof.LA.cm.dev.trl.txt'))
    print('no. of training trials', len(files_id_train))
    print('no. of validation trials', len(files_id_dev))

    train_set = Dataset_PartialSpoof_train(
        list_IDs=files_id_train,
        seglab=seglab_train,
        base_dir=os.path.join(args.database_path, 'train', 'con_wav'),
        args=args,
        algo=args.algo,
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
        num_workers=0, shuffle=True, drop_last=True)
    del train_set

    dev_set = Dataset_PartialSpoof_eval(
        list_IDs=files_id_dev,
        seglab=seglab_dev,
        base_dir=os.path.join(args.database_path, 'dev', 'con_wav'),
    )
    dev_loader = DataLoader(dev_set, batch_size=8, num_workers=0, shuffle=False)
    del dev_set

    # ---- Debug mode: limit batches for quick pipeline test ----
    debug_steps = args.debug_steps  # 0 = disabled, e.g. 5 = only 5 batches
    not_improving = 0
    epoch = 0
    bests = np.ones(n_mejores, dtype=float) * float('inf')
    best_loss = float('inf')

    if args.train:
        # NOTE: Do NOT pre-fill best_*.pth with np.savetxt placeholders
        # (that would overwrite real saved models on restart)
        while not_improving < args.num_epochs:
            print('######## Epoch {} ########'.format(epoch))
            train_epoch(train_loader, model, optimizer, device, epoch,
                        checkpoint_dir=model_save_path, debug_steps=debug_steps, args=args)
            val_loss, val_metrics = evaluate_accuracy(dev_loader, model, device, debug_steps=debug_steps)
            val_record = {
                'epoch': int(epoch),
                'val_loss': float(val_loss),
                'best_loss_before_epoch': float(best_loss) if np.isfinite(best_loss) else None,
                'metrics': val_metrics,
            }
            append_experiment_record(experiment_record_path, val_record)
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(),
                    os.path.join(model_save_path, 'best.pth'))
                print('New best epoch')
                not_improving = 0
            else:
                not_improving += 1
            for i in range(n_mejores):
                if bests[i] > val_loss:
                    for t in range(n_mejores - 1, i, -1):
                        bests[t] = bests[t - 1]
                        src = os.path.join(best_save_path,
                            'best_{}.pth'.format(t - 1))
                        dst = os.path.join(best_save_path,
                            'best_{}.pth'.format(t))
                        if os.path.exists(src):
                            shutil.move(src, dst)
                    bests[i] = val_loss
                    torch.save(model.state_dict(),
                        os.path.join(best_save_path,
                            'best_{}.pth'.format(i)))
                    break
            print('\n{} - val_loss={:.4f} | macro_f1={:.4f} | spoof_f1={:.4f} | bound_f1={:.4f}'.format(
                epoch, val_loss, val_metrics['macro_f1'], val_metrics['spoof_f1'], val_metrics['boundary_f1']))
            print('n-best losses:', bests)
            epoch += 1
            if epoch > 74:
                break
        print('Total epochs: ' + str(epoch))

    # ---- Final evaluation ----
    print('######## Eval ########')
    if args.average_model:
        actual_n = sum(
            1 for i in range(args.n_average_model)
            if os.path.exists(os.path.join(best_save_path,
                'best_{}.pth'.format(i)))
            and os.path.getsize(os.path.join(best_save_path,
                'best_{}.pth'.format(i))) > 1000)
        n_avg = min(args.n_average_model, actual_n)
        print('Averaging {} best models'.format(n_avg))

        if n_avg == 0:
            # No valid best_*.pth — fall back to best.pth or latest checkpoint
            best_single = os.path.join(model_save_path, 'best.pth')
            checkpoints = sorted(
                [f for f in os.listdir(model_save_path)
                 if f.startswith('checkpoint_') and f.endswith('.pth')],
                key=lambda x: os.path.getmtime(os.path.join(model_save_path, x))
            )
            if os.path.exists(best_single) and os.path.getsize(best_single) > 1000:
                print('Loading best.pth')
                model.load_state_dict(torch.load(best_single,
                    map_location=device))
            elif checkpoints:
                latest = os.path.join(model_save_path, checkpoints[-1])
                print('Loading latest checkpoint: {}'.format(latest))
                model.load_state_dict(torch.load(latest,
                    map_location=device))
            else:
                print('ERROR: No valid model found. Please train first.')
                sys.exit(1)
        else:
            model.load_state_dict(torch.load(
                os.path.join(best_save_path, 'best_0.pth'),
                map_location=device))
            sd = model.state_dict()
            for i in range(1, n_avg):
                model.load_state_dict(torch.load(
                    os.path.join(best_save_path, 'best_{}.pth'.format(i)),
                    map_location=device))
                sd2 = model.state_dict()
                for key in sd:
                    sd[key] = sd[key] + sd2[key]
            for key in sd:
                sd[key] = sd[key] / n_avg
            model.load_state_dict(sd)
    else:
        best_single = os.path.join(model_save_path, 'best.pth')
        checkpoints = sorted(
            [f for f in os.listdir(model_save_path)
             if f.startswith('checkpoint_') and f.endswith('.pth')],
            key=lambda x: os.path.getmtime(os.path.join(model_save_path, x))
        )
        if os.path.exists(best_single) and os.path.getsize(best_single) > 1000:
            model.load_state_dict(torch.load(best_single, map_location=device))
        elif checkpoints:
            latest = os.path.join(model_save_path, checkpoints[-1])
            print('Loading latest checkpoint: {}'.format(latest))
            model.load_state_dict(torch.load(latest, map_location=device))
        else:
            print('ERROR: No valid model found. Please train first.')
            sys.exit(1)

    tracks = 'LA' if args.algo == 5 else 'DF'
    if args.comment_eval:
        model_tag = model_tag + '_{}'.format(args.comment_eval)
    os.makedirs('./Scores/PartialSpoof', exist_ok=True)
    score_path = './Scores/PartialSpoof/{}.txt'.format(model_tag)
    if not os.path.exists(score_path):
        seglab_eval = load_seglab(
            os.path.join(args.database_path, 'segment_labels', 'dev_seglab_0.02.npy'))
        ps_proto_dir = os.path.join(args.protocols_path,
                                    'protocols', 'PartialSpoof_LA_cm_protocols')
        files_id_eval, _ = parse_ps_protocol(
            os.path.join(ps_proto_dir, 'PartialSpoof.LA.cm.dev.trl.txt'),
            is_eval=True)
        print('no. of eval trials (using dev set)', len(files_id_eval))
        eval_set = Dataset_PartialSpoof_eval(
            list_IDs=files_id_eval,
            seglab=seglab_eval,
            base_dir=os.path.join(args.database_path, 'dev', 'con_wav'),
        )
        produce_evaluation_file(eval_set, model, device, score_path)
        if args.export_analysis:
            export_analysis_artifacts(
                dataset=eval_set,
                model=model,
                device=device,
                export_dir=analysis_export_dir,
                max_batches=args.analysis_max_batches,
                batch_size=args.analysis_batch_size,
            )
    else:
        print('Score file already exists')
        if args.export_analysis:
            seglab_eval = load_seglab(
                os.path.join(args.database_path, 'segment_labels', 'dev_seglab_0.02.npy'))
            ps_proto_dir = os.path.join(args.protocols_path,
                                        'protocols', 'PartialSpoof_LA_cm_protocols')
            files_id_eval, _ = parse_ps_protocol(
                os.path.join(ps_proto_dir, 'PartialSpoof.LA.cm.dev.trl.txt'),
                is_eval=True)
            eval_set = Dataset_PartialSpoof_eval(
                list_IDs=files_id_eval,
                seglab=seglab_eval,
                base_dir=os.path.join(args.database_path, 'dev', 'con_wav'),
            )
            export_analysis_artifacts(
                dataset=eval_set,
                model=model,
                device=device,
                export_dir=analysis_export_dir,
                max_batches=args.analysis_max_batches,
                batch_size=args.analysis_batch_size,
            )
