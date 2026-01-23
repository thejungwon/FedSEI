#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Benchmark plain PyTorch vs CrypTen (single-process) inference latency
for custom CNN models (small / medium / large).

Usage examples:

    # Single run
    python benchmark_crypten_cnn.py --model small --batch-size 32 --device cpu

    # Full sweep (small/medium/large x batch 1..64, 3 repeats each)
    python benchmark_crypten_cnn.py --sweep --device cpu
"""

import argparse
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import crypten
import crypten.nn.onnx_converter as onnx_conv

# force CrypTen to use opset 11
onnx_conv._OPSET_VERSION = 11


# =========================
# 1. CNN model definitions
# =========================


class SmallCNN(nn.Module):
    """
    Small CNN:
    - Conv(3 -> 32) -> ReLU -> MaxPool
    - Conv(32 -> 64) -> ReLU -> MaxPool
    - FC(64*8*8 -> 256) -> ReLU
    - FC(256 -> num_classes)
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # input: (B, 3, 32, 32)
        x = F.relu(self.conv1(x))  # (B, 32, 32, 32)
        x = self.pool(x)  # (B, 32, 16, 16)
        x = F.relu(self.conv2(x))  # (B, 64, 16, 16)
        x = self.pool(x)  # (B, 64, 8, 8)
        x = torch.flatten(x, 1)  # (B, 64*8*8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MediumCNN(nn.Module):
    """
    Medium CNN:
    - Conv(3 -> 32) -> ReLU -> MaxPool
    - Conv(32 -> 64) -> ReLU
    - Conv(64 -> 128) -> ReLU -> MaxPool
    - FC(128*8*8 -> 512) -> ReLU
    - FC(512 -> num_classes)
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # (B, 32, 32, 32)
        x = self.pool(x)  # (B, 32, 16, 16)
        x = F.relu(self.conv2(x))  # (B, 64, 16, 16)
        x = F.relu(self.conv3(x))  # (B, 128, 16, 16)
        x = self.pool(x)  # (B, 128, 8, 8)
        x = torch.flatten(x, 1)  # (B, 128*8*8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class LargeCNN(nn.Module):
    """
    Large CNN:
    - Conv(3 -> 64) -> ReLU
    - Conv(64 -> 64) -> ReLU -> MaxPool
    - Conv(64 -> 128) -> ReLU
    - Conv(128 -> 128) -> ReLU -> MaxPool
    - FC(128*8*8 -> 512) -> ReLU
    - FC(512 -> 256) -> ReLU
    - FC(256 -> num_classes)
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # (B, 64, 32, 32)
        x = F.relu(self.conv2(x))  # (B, 64, 32, 32)
        x = self.pool(x)  # (B, 64, 16, 16)
        x = F.relu(self.conv3(x))  # (B, 128, 16, 16)
        x = F.relu(self.conv4(x))  # (B, 128, 16, 16)
        x = self.pool(x)  # (B, 128, 8, 8)
        x = torch.flatten(x, 1)  # (B, 128*8*8)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

        # =========================
        # 2. Utils - model/input creation and benchmarking
        # =========================


def build_cnn(model_name: str, num_classes: int = 10) -> nn.Module:
    model_name = model_name.lower()
    if model_name == "small":
        return SmallCNN(num_classes=num_classes)
    elif model_name == "medium":
        return MediumCNN(num_classes=num_classes)
    elif model_name == "large":
        return LargeCNN(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def build_dummy_input(batch_size: int) -> torch.Tensor:
    # CIFAR-style: (B, 3, 32, 32)
    return torch.randn(batch_size, 3, 32, 32)


def benchmark_plain_inference(
    model: nn.Module,
    dummy_input: torch.Tensor,
    iters: int = 30,
    warmup: int = 5,
    device: str = "cpu",
) -> float:
    """PyTorch plaintext inference latency (sec/batch)."""
    model = model.to(device)
    model.eval()
    dummy_input = dummy_input.to(device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    start = time.time()
    with torch.no_grad():
        for _ in range(iters):
            _ = model(dummy_input)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    end = time.time()

    return (end - start) / iters


def benchmark_crypten_inference_single(
    model: nn.Module,
    dummy_input: torch.Tensor,
    iters: int = 10,
    warmup: int = 3,
    device: str = "cpu",
) -> float:
    """
    CrypTen (single-process) inference latency (sec/batch).
    world_size=1 is not real SMPC; used to measure cryptensor overhead.
    Assumes crypten.init() is called once in main.
    """
    # Align device
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            print(
                "[CrypTen] CUDA requested but not available. Falling back to CPU.",
                flush=True,
            )
            device = "cpu"

    model = model.to(device)
    dummy_input = dummy_input.to(device)
    model.eval()

    # (Optional) align CrypTen default device
    if device.startswith("cuda"):
        try:
            crypten.cuda.set_default_device(device)
        except Exception as e:
            print(
                f"[CrypTen] Warning: failed to set default CUDA device: {e}", flush=True
            )

            # PyTorch -> CrypTen model conversion
    try:
        crypten_model = crypten.nn.from_pytorch(model, dummy_input)
    except ValueError as e:
        print(f"[CrypTen] from_pytorch failed for this model: {e}", flush=True)
        print(
            "This model uses ops not supported by CrypTen's ONNX converter.", flush=True
        )
        return float("nan")

    for module in crypten_model.modules():
        if hasattr(module, "alpha") and isinstance(module.alpha, float):
            module.alpha = int(module.alpha)
        if hasattr(module, "beta") and isinstance(module.beta, float):
            module.beta = int(module.beta)

    crypten_model.encrypt()  # secret-shared model (single process)

    # Convert input to CrypTen tensor
    x_enc = crypten.cryptensor(dummy_input)

    # Warmup
    for _ in range(warmup):
        _ = crypten_model(x_enc)

    start = time.time()
    for _ in range(iters):
        _ = crypten_model(x_enc)
    end = time.time()

    return (end - start) / iters

    # =========================
    # 3. Sweep utilities
    # =========================


def run_single_experiment(
    model_name: str,
    batch_size: int,
    device: str,
    plain_iters: int,
    plain_warmup: int,
    mpc_iters: int,
    mpc_warmup: int,
) -> Tuple[float, float]:
    """Return one plain / CrypTen latency measurement (sec/batch)."""
    model = build_cnn(model_name, num_classes=10)
    dummy_input = build_dummy_input(batch_size)

    plain_latency = benchmark_plain_inference(
        model,
        dummy_input,
        iters=plain_iters,
        warmup=plain_warmup,
        device=device,
    )

    mpc_latency = benchmark_crypten_inference_single(
        model,
        dummy_input,
        iters=mpc_iters,
        warmup=mpc_warmup,
        device=device,
    )

    return plain_latency, mpc_latency


def run_sweep(
    device: str,
    batch_sizes=None,
    models=None,
    repeats: int = 3,
    plain_iters: int = 30,
    plain_warmup: int = 5,
    mpc_iters: int = 10,
    mpc_warmup: int = 3,
):
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    if models is None:
        models = ["small", "medium", "large"]

    results = {m: {} for m in models}

    for model_name in models:
        print(f"\n=== Model: {model_name} ===", flush=True)
        for bs in batch_sizes:
            plain_list = []
            mpc_list = []
            for r in range(repeats):
                plain_lat, mpc_lat = run_single_experiment(
                    model_name=model_name,
                    batch_size=bs,
                    device=device,
                    plain_iters=plain_iters,
                    plain_warmup=plain_warmup,
                    mpc_iters=mpc_iters,
                    mpc_warmup=mpc_warmup,
                )
                plain_list.append(plain_lat * 1000.0)  # convert to ms
                if mpc_lat == mpc_lat:  # not NaN
                    mpc_list.append(mpc_lat * 1000.0)

            plain_mean = float(np.mean(plain_list))
            plain_std = float(np.std(plain_list))

            if len(mpc_list) > 0:
                mpc_mean = float(np.mean(mpc_list))
                mpc_std = float(np.std(mpc_list))
            else:
                mpc_mean, mpc_std = float("nan"), float("nan")

            results[model_name][bs] = {
                "plain_mean": plain_mean,
                "plain_std": plain_std,
                "mpc_mean": mpc_mean,
                "mpc_std": mpc_std,
            }

            print(
                f"  batch={bs:3d} | "
                f"Plain: {plain_mean:.3f} ± {plain_std:.3f} ms | "
                f"CrypTen-1P: "
                f"{'nan' if np.isnan(mpc_mean) else f'{mpc_mean:.3f} ± {mpc_std:.3f} ms'}",
                flush=True,
            )


# =========================
# 4. main
# =========================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="small", choices=["small", "medium", "large"]
    )
    parser.add_argument("--batch-size", type=int, default=32)

    parser.add_argument("--plain-iters", type=int, default=30)
    parser.add_argument("--plain-warmup", type=int, default=5)
    parser.add_argument("--mpc-iters", type=int, default=10)
    parser.add_argument("--mpc-warmup", type=int, default=3)

    parser.add_argument(
        "--device", type=str, default="cpu", help="cpu or cuda[:idx], e.g., cuda:0"
    )

    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run sweep over models={small,medium,large} and batch sizes "
        "[1,2,4,8,16,32,64], with repeats.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of repetitions for each (model, batch_size) in sweep mode.",
    )

    args = parser.parse_args()

    # Check CUDA availability
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print(f"[Warning] CUDA not available. Falling back to CPU.", flush=True)
        args.device = "cpu"

    # Initialize CrypTen once
    print("[Init] Calling crypten.init()...", flush=True)
    crypten.init()
    print("[Init] crypten.init() done.", flush=True)

    if args.sweep:
        # Full sweep mode
        run_sweep(
            device=args.device,
            batch_sizes=[1, 2, 4, 8, 16, 32, 64],
            models=["small", "medium", "large"],
            repeats=args.repeats,
            plain_iters=args.plain_iters,
            plain_warmup=args.plain_warmup,
            mpc_iters=args.mpc_iters,
            mpc_warmup=args.mpc_warmup,
        )
    else:
        # Single-run mode
        model = build_cnn(args.model, num_classes=10)
        dummy_input = build_dummy_input(args.batch_size)

        plain_latency = benchmark_plain_inference(
            model,
            dummy_input,
            iters=args.plain_iters,
            warmup=args.plain_warmup,
            device=args.device,
        )
        print(
            f"[Plain]  model={args.model:6s}, batch={args.batch_size:3d}, "
            f"device={args.device} → {plain_latency*1000:.2f} ms / batch",
            flush=True,
        )

        mpc_latency = benchmark_crypten_inference_single(
            model,
            dummy_input,
            iters=args.mpc_iters,
            warmup=args.mpc_warmup,
            device=args.device,
        )
        if mpc_latency == mpc_latency:  # not NaN
            print(
                f"[CrypTen-1P] model={args.model:6s}, batch={args.batch_size:3d}, "
                f"device={args.device} → {mpc_latency*1000:.2f} ms / batch",
                flush=True,
            )
        else:
            print(
                f"[CrypTen-1P] model={args.model:6s} → unsupported by from_pytorch.",
                flush=True,
            )


if __name__ == "__main__":
    main()
