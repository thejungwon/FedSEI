#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Full E2E SMPC Ensemble Demo.

Scenario:
  1. Data (x) is encrypted immediately (Secret Shared).
  2. N Models are encrypted (Weights are Secret Shared).
  3. Inference (Forward pass) happens entirely in MPC.
  4. Aggregation (Voting) happens in MPC.

Everything is processed as CrypTensors. No plaintext until the final result.
"""

import argparse
import os
import random
import time
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import crypten
import crypten.mpc as mpc
import crypten.communicator as comm
import crypten.nn.onnx_converter as onnx_conv

# force CrypTen to use opset 11
onnx_conv._OPSET_VERSION = 11

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "3"))
PRECISION_BITS = int(
    os.environ.get("PRECISION_BITS", "16")
)  # Keep modest to avoid overflow.
SOFTMAX_TEMP = 1.0
TTA_STEPS = int(os.environ.get("TTA_STEPS", "0"))
TTA_NOISE_STD = float(os.environ.get("TTA_NOISE_STD", "0.05"))
TTA_FLIP_PROB = float(os.environ.get("TTA_FLIP_PROB", "0.5"))
TTA_GAMMA = float(os.environ.get("TTA_GAMMA", "1.0"))
TTA_EPS = 1e-4


# Conv2d under SMPC is slow, so use a tiny model.
class TinyCNN(nn.Module):
    def __init__(self, in_ch: int, num_classes: int):
        super().__init__()
        # Tiny structure to minimize compute cost.
        self.conv1 = nn.Conv2d(in_ch, 4, kernel_size=3, padding=1)
        self.fc = nn.Linear(4 * 8 * 8, num_classes)

    def forward(self, x):
        # CrypTen supports ReLU and flatten.
        out = self.conv1(x)
        out = F.relu(out)
        out = out.flatten(1)
        out = self.fc(out)
        return out


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)


def make_dummy_data(seed, num_samples, in_ch, img_size, num_classes):
    set_seed(seed)
    x = torch.randn(num_samples, in_ch, img_size, img_size)
    y = torch.randint(0, num_classes, (num_samples,))
    return x, y


def tta_augment_plain(x: torch.Tensor) -> torch.Tensor:
    x_aug = x.clone()
    if TTA_NOISE_STD > 0:
        x_aug = x_aug + TTA_NOISE_STD * torch.randn_like(x_aug)
    if TTA_FLIP_PROB > 0 and random.random() < TTA_FLIP_PROB:
        x_aug = torch.flip(x_aug, dims=[3])
    return x_aug


# --- Secure Voting Logic (inputs/outputs are encrypted) ---
# Assumes inputs are always CrypTensors.


def secure_hard_voting(enc_logits_k_nc):
    enc_preds = enc_logits_k_nc.argmax(dim=-1)
    num_classes = enc_logits_k_nc.size(-1)
    vote_list = []
    for c in range(num_classes):
        vote_list.append((enc_preds == c).sum(dim=0))
    enc_votes = crypten.stack(vote_list, dim=0)
    return enc_votes.argmax(dim=0, one_hot=False)


def secure_soft_voting(enc_probs_k_nc):
    enc_mean = enc_probs_k_nc.mean(dim=0)
    return enc_mean, enc_mean.argmax(dim=-1, one_hot=False)


def secure_entropy_voting(enc_probs_k_nc):
    eps = 1e-4  # Adjust epsilon for MPC precision.
    enc_log = (enc_probs_k_nc + eps).log()
    enc_ent = -(enc_probs_k_nc * enc_log).sum(dim=-1)
    enc_w = (-enc_ent).exp()
    enc_w_norm = enc_w / (enc_w.sum(dim=0, keepdim=True) + eps)
    enc_weighted = (enc_w_norm.unsqueeze(-1) * enc_probs_k_nc).sum(dim=0)
    return enc_weighted, enc_weighted.argmax(dim=-1, one_hot=False)


def secure_spectral_voting(enc_probs_k_nc, num_iters=2):
    # SMPC power iteration.
    K = enc_probs_k_nc.size(0)
    eps = 1e-4

    # Argmax can be expensive; keep the original argmax-based flow here.
    enc_one_hot = enc_probs_k_nc.argmax(dim=-1, one_hot=True)
    enc_conf = (enc_probs_k_nc * enc_one_hot).sum(dim=-1)  # [K, N]

    enc_centered = enc_conf - enc_conf.mean(dim=1, keepdim=True)
    enc_cov = enc_centered.matmul(enc_centered.transpose(0, 1))

    # Init vector (all ones), encrypted.
    enc_v = crypten.cryptensor(torch.ones(K), precision=PRECISION_BITS)

    for _ in range(num_iters):
        enc_v = enc_cov.matmul(enc_v)
        enc_v = enc_v / (enc_v.sum() + eps)  # Normalize inside loop to prevent overflow

    enc_alpha = enc_v / (enc_v.sum() + eps)
    enc_weighted = (enc_alpha.view(-1, 1, 1) * enc_probs_k_nc).sum(dim=0)
    return enc_weighted, enc_weighted.argmax(dim=-1, one_hot=False)


def mpc_main():
    crypten.init()
    rank = comm.get().get_rank()
    world_size = comm.get().get_world_size()

    # Keep the demo tiny (full SMPC is slow).
    # [Config] Image size: 8x8, samples: 4.
    N_SAMPLES = 10
    IMG_SIZE = 8
    IN_CH = 1
    N_CLASSES = 3

    if rank == 0:
        print(f"--- Full E2E SMPC Ensemble Demo (World Size: {world_size}) ---")
        print(f"Data: {N_SAMPLES} samples, {IN_CH}x{IMG_SIZE}x{IMG_SIZE}")
        print("Status: Preparing Encrypted Data & Models...\n")

    # ---------------------------------------------------------
    # 1. Data Encryption (Input x -> Encrypted x)
    # ---------------------------------------------------------
    # Assume rank 0 owns the data.
    x_plain, y_plain = make_dummy_data(42, N_SAMPLES, IN_CH, IMG_SIZE, N_CLASSES)

    # Only rank 0 provides real data; others provide dummy tensors.
    # src=0 means rank 0's data is secret-shared to all parties.
    enc_x = crypten.cryptensor(x_plain, src=0, precision=PRECISION_BITS)

    if rank == 0:
        print("[Step 1] Input data 'x' is now encrypted (secret shared).")

    # ---------------------------------------------------------
    # 2. Model Encryption (PyTorch Model -> CrypTen Encrypted Model)
    # ---------------------------------------------------------
    # Assume each rank owns one model (Model_i).
    # Build a list of K (= world_size) encrypted models.

    enc_models = []

    # Dummy input for tracing the graph.
    dummy_input = torch.empty(1, IN_CH, IMG_SIZE, IMG_SIZE)

    for i in range(world_size):
        # 1) Build the plaintext model structure (all nodes know the graph).
        raw_model = TinyCNN(IN_CH, N_CLASSES)

        # 2) Load weights: assume rank i owns model i.
        if rank == i:
            # Only rank i initializes its own weights (seeded).
            set_seed(100 + i)
            # (Re-init weights specifically for this rank)
            for m in raw_model.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        # 3) Convert to CrypTen model.
        # from_pytorch traces the model graph.
        enc_model_i = crypten.nn.from_pytorch(raw_model, dummy_input)

        # 4) Encrypt weights (.encrypt).
        # src=i: rank i secret-shares its weights to all parties.
        enc_model_i.encrypt(src=i)
        enc_models.append(enc_model_i)

        if rank == 0:
            print(f"  - Model {i} Encrypted (Owner: Rank {i})")

    if rank == 0:
        print(f"[Step 2] {len(enc_models)} Models are fully encrypted.\n")
        print("[Step 3] Starting E2E encrypted inference (forward pass)...")

    # ---------------------------------------------------------
    # 3. Encrypted Inference (Forward Pass in MPC)
    # ---------------------------------------------------------
    # enc_model(enc_x) -> enc_logits
    # All layer ops run under SMPC.

    enc_logits_list = []
    enc_probs_list = []

    for i, enc_m in enumerate(enc_models):
        t0 = time.time()

        # Feed encrypted input into encrypted model.
        enc_logits = enc_m(enc_x)

        # Logits -> Probs (Softmax) in MPC
        # CrypTen softmax is approximate and expensive.
        enc_probs = enc_logits.softmax(dim=-1)

        enc_logits_list.append(enc_logits)
        enc_probs_list.append(enc_probs)

        t_elapsed = time.time() - t0
        if rank == 0:
            print(f"  > Model {i} Inference Complete ({t_elapsed:.2f}s)")

    # Stack results: [K, N, C]
    enc_logits_k_nc = crypten.stack(enc_logits_list, dim=0)
    enc_probs_k_nc = crypten.stack(enc_probs_list, dim=0)

    if rank == 0:
        print(f"[Step 3] Inference Done. Encrypted Logits Stacked.\n")
        print("[Step 4] Starting Secure Voting...\n")

    # ---------------------------------------------------------
    # 4. Secure Ensemble Voting
    # ---------------------------------------------------------

    # (A) Hard Voting
    t0 = time.time()
    enc_hard_pred = secure_hard_voting(enc_logits_k_nc)
    if rank == 0:
        print(f"  - Hard Voting Done ({time.time()-t0:.3f}s)")

    # (B) Soft Voting
    t0 = time.time()
    enc_soft_prob, enc_soft_pred = secure_soft_voting(enc_probs_k_nc)
    if rank == 0:
        print(f"  - Soft Voting Done ({time.time()-t0:.3f}s)")

    # (C) Entropy Weighted
    t0 = time.time()
    enc_ent_prob, enc_ent_pred = secure_entropy_voting(enc_probs_k_nc)
    if rank == 0:
        print(f"  - Entropy Voting Done ({time.time()-t0:.3f}s)")

    # (D) Spectral
    t0 = time.time()
    enc_spec_prob, enc_spec_pred = secure_spectral_voting(enc_probs_k_nc)
    if rank == 0:
        print(f"  - Spectral Voting Done ({time.time()-t0:.3f}s)")

    # ---------------------------------------------------------
    # 4.1 TTA-based L2 Weighted Ensemble (Optional)
    # ---------------------------------------------------------
    enc_tta_soft_pred = None
    enc_tta_l2_pred = None

    if TTA_STEPS > 0:
        if rank == 0:
            print(f"[Step 4.1] Starting TTA L2-weighted ensemble (T={TTA_STEPS})...\n")
        tta_t0 = time.time()

        K = enc_logits_k_nc.size(0)
        N = enc_logits_k_nc.size(1)
        C = enc_logits_k_nc.size(2)

        sum_probs_k_nc = crypten.cryptensor(
            torch.zeros(K, N, C), src=0, precision=PRECISION_BITS
        )
        sum_l2_k_n = crypten.cryptensor(
            torch.zeros(K, N), src=0, precision=PRECISION_BITS
        )

        for _ in range(TTA_STEPS):
            if rank == 0:
                x_aug_plain = tta_augment_plain(x_plain)
            else:
                x_aug_plain = torch.zeros_like(x_plain)

            enc_x_aug = crypten.cryptensor(x_aug_plain, src=0, precision=PRECISION_BITS)

            enc_probs_t_list = []
            enc_l2_t_list = []
            for i, enc_m in enumerate(enc_models):
                enc_logits_t = enc_m(enc_x_aug)
                enc_probs_t = enc_logits_t.softmax(dim=-1)
                enc_probs_t_list.append(enc_probs_t)

                diff = enc_logits_t - enc_logits_list[i]
                l2 = diff.pow(2).sum(dim=-1).sqrt()
                enc_l2_t_list.append(l2)

            enc_probs_t_k_nc = crypten.stack(enc_probs_t_list, dim=0)
            enc_l2_t_k_n = crypten.stack(enc_l2_t_list, dim=0)

            sum_probs_k_nc = sum_probs_k_nc + enc_probs_t_k_nc
            sum_l2_k_n = sum_l2_k_n + enc_l2_t_k_n

        enc_mean_probs_k_nc = sum_probs_k_nc / float(TTA_STEPS)
        enc_l2_k_n = sum_l2_k_n / float(TTA_STEPS)

        enc_tta_soft_prob = enc_mean_probs_k_nc.mean(dim=0)
        enc_tta_soft_pred = enc_tta_soft_prob.argmax(dim=-1, one_hot=False)

        enc_w_k_n = (enc_l2_k_n * TTA_GAMMA).exp()
        enc_w_k_n = enc_w_k_n / (enc_w_k_n.sum(dim=0, keepdim=True) + TTA_EPS)
        enc_weighted = (enc_w_k_n.unsqueeze(-1) * enc_mean_probs_k_nc).sum(dim=0)
        enc_tta_l2_pred = enc_weighted.argmax(dim=-1, one_hot=False)
        if rank == 0:
            print(f"[Step 4.1] TTA completed ({time.time()-tta_t0:.3f}s)")

    # ---------------------------------------------------------
    # 5. Decrypt Final Result Only
    # ---------------------------------------------------------
    # Only the final result is decrypted. Model weights and per-model outputs stay hidden.

    final_preds = enc_ent_pred.get_plain_text().long()
    labels = y_plain.long()

    acc = (final_preds == labels).float().mean().item() * 100.0

    tta_soft_preds = None
    tta_l2_preds = None
    if enc_tta_soft_pred is not None:
        tta_soft_preds = enc_tta_soft_pred.get_plain_text().long()
    if enc_tta_l2_pred is not None:
        tta_l2_preds = enc_tta_l2_pred.get_plain_text().long()

    if rank == 0:
        print(f"\n[Result] Final Entropy Ensemble Accuracy: {acc:.2f}%")
        print(
            "Note: All computations (Input->Conv->ReLU->Voting) were performed in SMPC."
        )
        if tta_soft_preds is not None:
            tta_soft_acc = (tta_soft_preds == labels).float().mean().item() * 100.0
            print(f"[Result] TTA Soft Ensemble Accuracy: {tta_soft_acc:.2f}%")
        if tta_l2_preds is not None:
            tta_l2_acc = (tta_l2_preds == labels).float().mean().item() * 100.0
            print(f"[Result] TTA L2-weighted Accuracy: {tta_l2_acc:.2f}%")


if __name__ == "__main__":
    wrapped = mpc.run_multiprocess(world_size=WORLD_SIZE)(mpc_main)
    wrapped()
