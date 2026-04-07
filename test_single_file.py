#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Lightweight CPU-first forward-shape check for model outputs."""

import numpy as np
import torch

from model import Model


def test_single_forward():
    class Args:
        emb_size = 144
        num_encoders = 12
        num_classes = 3

    args = Args()

    # Prefer CPU to avoid disturbing other running GPU jobs.
    device = 'cpu'
    print(f"Device: {device}")

    print("Loading model...")
    model = Model(args, device).to(device)
    model.eval()
    print("Model loaded successfully")

    # Fixed-length dummy waveform matching the training pipeline.
    x_inp = torch.tensor(np.zeros((1, 66800), dtype=np.float32), device=device)
    print(f"Input tensor shape: {x_inp.shape}")

    with torch.no_grad():
        p_bound_logits, logits_dia, h_prime = model(x_inp)

    print(f"p_bound_logits shape: {tuple(p_bound_logits.shape)}")
    print(f"logits_dia shape: {tuple(logits_dia.shape)}")
    print(f"h_prime shape: {tuple(h_prime.shape)}")
    print("\n✓ Output interface check passed.")


if __name__ == '__main__':
    test_single_forward()

