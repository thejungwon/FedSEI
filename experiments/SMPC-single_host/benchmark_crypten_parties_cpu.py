#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Benchmark plain PyTorch vs CrypTen inference latency
for SmallCNN / SmallMLP on CPU, with different number of parties (world_size).

Usage examples (CPU only):

    # CNN, 1-party (no real MPC, CrypTen overhead only)
    python benchmark_crypten_parties_cpu.py --model cnn --batch-size 32 --world-size 1

    # CNN, 3-party MPC
    python benchmark_crypten_parties_cpu.py --model cnn --batch-size 32 --world-size 3

    # MLP, 1-party
    python benchmark_crypten_parties_cpu.py --model mlp --batch-size 32 --world-size 1

    # MLP, 3-party
    python benchmark_crypten_parties_cpu.py --model mlp --batch-size 32 --world-size 3

    # Sweep over world_size × batch_size for cnn + mlp
    python benchmark_crypten_parties_cpu.py --sweep
"""

import argparse
import time
from typing import Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

import crypten
import crypten.mpc as mpc

import crypten.nn.onnx_converter as onnx_conv

# force CrypTen to use opset 11
onnx_conv._OPSET_VERSION = 11


import multiprocessing as mp
import numpy as np

# Needed on macOS for nested functions + run_multiprocess
mp.set_start_method("fork", force=True)

# Limit PyTorch CPU threads (avoid excessive multithreading)
torch.set_num_threads(1)


# =========================
# 1. Model definitions: SmallCNN / SmallMLP
# =========================


class SmallCNN(nn.Module):
    """
    Small CNN for CIFAR-like 32x32x3 input:
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
        # x: (B, 3, 32, 32)
        x = F.relu(self.conv1(x))
        x = self.pool(x)  # (B, 32, 16, 16)
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # (B, 64, 8, 8)
        x = torch.flatten(x, 1)  # (B, 64*8*8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SmallMLP(nn.Module):
    """
    Simple MLP for 32x32x3 input (flattened):
    - FC(3*32*32 -> 512) -> ReLU
    - FC(512 -> num_classes)
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # x: (B, 3, 32, 32)
        x = x.view(x.size(0), -1)  # (B, 3*32*32)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def create_model(model_type: str, num_classes: int = 10) -> nn.Module:
    model_type = model_type.lower()
    if model_type == "cnn":
        return SmallCNN(num_classes=num_classes)
    elif model_type == "mlp":
        return SmallMLP(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model_type: {model_type} (use 'cnn' or 'mlp')")


def model_name_for_log(model_type: str) -> str:
    return "SmallCNN" if model_type.lower() == "cnn" else "SmallMLP"

    # =========================
    # 2. Utils - input & plain benchmark
    # =========================


def build_dummy_input(batch_size: int) -> torch.Tensor:
    # CIFAR-style: (B, 3, 32, 32) on CPU
    return torch.randn(batch_size, 3, 32, 32)


def benchmark_plain_inference(
    model_type: str,
    batch_size: int,
    iters: int = 30,
    warmup: int = 5,
) -> float:
    """
    Plain PyTorch inference latency (sec/batch, CPU only).
    """
    torch.set_num_threads(1)

    device = torch.device("cpu")
    model = create_model(model_type, num_classes=10).to(device)
    model.eval()

    dummy_input = build_dummy_input(batch_size).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

            # Timing
    start = time.time()
    with torch.no_grad():
        for _ in range(iters):
            _ = model(dummy_input)
    end = time.time()

    return (end - start) / iters

    # =========================
    # 3. CrypTen 1-party (single process) benchmark
    # =========================


def benchmark_crypten_singleparty(
    model_type: str,
    batch_size: int,
    iters: int = 10,
    warmup: int = 3,
) -> float:
    """
    CrypTen (world_size=1, single-process) inference latency (sec/batch, CPU only).
    Not real MPC; used to measure CrypTen overhead.
    """
    torch.set_num_threads(1)
    print("[Init] Calling crypten.init() (single-party)...", flush=True)
    crypten.init()
    print("[Init] crypten.init() done (single-party).", flush=True)

    device = torch.device("cpu")
    model = create_model(model_type, num_classes=10).to(device)
    model.eval()

    dummy_input = build_dummy_input(batch_size).to(device)

    # PyTorch -> CrypTen model conversion
    try:
        crypten_model = crypten.nn.from_pytorch(model, dummy_input)
    except ValueError as e:
        print(f"[CrypTen-1P] ({model_type}) from_pytorch failed: {e}", flush=True)
        print(
            "This model likely uses ops not supported by CrypTen's ONNX converter.",
            flush=True,
        )
        return float("nan")

    for module in crypten_model.modules():
        if hasattr(module, "alpha") and isinstance(module.alpha, float):
            module.alpha = int(module.alpha)
        if hasattr(module, "beta") and isinstance(module.beta, float):
            module.beta = int(module.beta)
    crypten_model.encrypt()
    x_enc = crypten.cryptensor(dummy_input)

    # Warmup
    for _ in range(warmup):
        _ = crypten_model(x_enc)

        # Timing
    start = time.time()
    for _ in range(iters):
        _ = crypten_model(x_enc)
    end = time.time()

    return (end - start) / iters

    # =========================
    # 4. CrypTen multi-party (2P / 3P...) benchmark - CPU only
    # =========================


def benchmark_crypten_multiparty(
    model_type: str,
    world_size: int,
    batch_size: int,
    iters: int = 10,
    warmup: int = 3,
) -> float:
    """
    CrypTen multi-process (P-party) inference latency (CPU only).
    - Use @mpc.run_multiprocess(world_size=...)
    - Define worker without args, use comm.get().get_rank() inside
    - Rank 0 returns latency; others return None
    """
    assert world_size >= 2, "Use benchmark_crypten_singleparty for world_size=1."

    model_type = model_type.lower()

    @mpc.run_multiprocess(world_size=world_size)
    def _mpc_worker():
        # Initialize CrypTen + PyTorch per process
        print("[Init] Calling crypten.init() (multiparty worker)...", flush=True)
        crypten.init()
        print("[Init] crypten.init() done (multiparty worker).", flush=True)
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except AttributeError:
            pass

        comm = crypten.comm.get()
        rank = comm.get_rank()
        ws = comm.get_world_size()

        device = torch.device("cpu")

        # Build model and input
        model = create_model(model_type, num_classes=10).to(device)
        model.eval()
        dummy_input = build_dummy_input(batch_size).to(device)

        # PyTorch -> CrypTen model conversion
        try:
            crypten_model = crypten.nn.from_pytorch(model, dummy_input)
        except ValueError as e:
            if rank == 0:
                print(
                    f"[CrypTen-{ws}P] ({model_type}) from_pytorch failed: {e}",
                    flush=True,
                )
                print(
                    "This model likely uses ops not supported by CrypTen's ONNX converter.",
                    flush=True,
                )
            comm.barrier()
            # On failure, rank 0 returns NaN and others return None
            return float("nan") if rank == 0 else None

            # Encrypt
        crypten_model.encrypt()
        x_enc = crypten.cryptensor(dummy_input)

        # Warmup
        for _ in range(warmup):
            _ = crypten_model(x_enc)
        comm.barrier()

        # Time only on rank 0
        if rank == 0:
            start = time.time()
        for _ in range(iters):
            _ = crypten_model(x_enc)
        comm.barrier()
        if rank == 0:
            end = time.time()
            latency = (end - start) / iters
            return latency
        else:
            return None

            # Run: returns list of rank results

    results = _mpc_worker()

    # results is a list of length world_size
    if isinstance(results, list):
        for v in results:
            if v is not None:
                return float(v)
        return float("nan")
    else:
        # Fallback for unexpected single value
        return float(results)

        # =========================
        # 5. Sweep utilities
        # =========================


def run_single_experiment(
    model_type: str,
    world_size: int,
    batch_size: int,
    plain_iters: int,
    plain_warmup: int,
    mpc_iters: int,
    mpc_warmup: int,
) -> Tuple[float, float]:
    """
    (plain_latency, mpc_latency) in seconds per batch.
    """
    plain_latency = benchmark_plain_inference(
        model_type=model_type,
        batch_size=batch_size,
        iters=plain_iters,
        warmup=plain_warmup,
    )

    if world_size == 1:
        mpc_latency = benchmark_crypten_singleparty(
            model_type=model_type,
            batch_size=batch_size,
            iters=mpc_iters,
            warmup=mpc_warmup,
        )
    else:
        mpc_latency = benchmark_crypten_multiparty(
            model_type=model_type,
            world_size=world_size,
            batch_size=batch_size,
            iters=mpc_iters,
            warmup=mpc_warmup,
        )

    return plain_latency, mpc_latency


def parse_int_list(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def run_sweep(
    models: List[str],
    world_sizes: List[int],
    batch_sizes: List[int],
    repeats: int,
    plain_iters: int,
    plain_warmup: int,
    mpc_iters: int,
    mpc_warmup: int,
):
    """
    Sweep over (model_type, world_size, batch_size).
    Print mean ± std (ms) to stdout
    per model and batch size.
    """
    # results[model_type][world_size][batch_size] = dict(...)
    results: Dict[str, Dict[int, Dict[int, Dict[str, float]]]] = {}

    print("\n========== SWEEP START ==========", flush=True)
    print(f"Models      : {models}", flush=True)
    print(f"World sizes : {world_sizes}", flush=True)
    print(f"Batch sizes : {batch_sizes}", flush=True)
    print(f"Repeats     : {repeats}", flush=True)
    print("=================================\n", flush=True)

    for model_type in models:
        mname = model_name_for_log(model_type)
        results[model_type] = {}
        print(f"\n===== Model: {mname} ({model_type}) =====", flush=True)
        for ws in world_sizes:
            results[model_type][ws] = {}
            print(f"\n  -- World size: {ws} --", flush=True)
            for bs in batch_sizes:
                plain_list = []
                mpc_list = []

                for r in range(repeats):
                    plain_lat, mpc_lat = run_single_experiment(
                        model_type=model_type,
                        world_size=ws,
                        batch_size=bs,
                        plain_iters=plain_iters,
                        plain_warmup=plain_warmup,
                        mpc_iters=mpc_iters,
                        mpc_warmup=mpc_warmup,
                    )
                    plain_list.append(plain_lat * 1000.0)  # ms
                    mpc_list.append(mpc_lat * 1000.0)  # ms

                plain_mean = float(np.mean(plain_list))
                plain_std = float(np.std(plain_list))
                mpc_mean = float(np.mean(mpc_list))
                mpc_std = float(np.std(mpc_list))

                results[model_type][ws][bs] = {
                    "plain_mean": plain_mean,
                    "plain_std": plain_std,
                    "mpc_mean": mpc_mean,
                    "mpc_std": mpc_std,
                }

                print(
                    f"    batch={bs:3d} | "
                    f"Plain: {plain_mean:.3f} ± {plain_std:.3f} ms | "
                    f"CrypTen-{ws}P: {mpc_mean:.3f} ± {mpc_std:.3f} ms",
                    flush=True,
                )

    print("\n========== SWEEP DONE ==========\n", flush=True)

    return results

    # =========================
    # 6. main
    # =========================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="cnn",
        choices=["cnn", "mlp"],
        help="which model to benchmark: cnn or mlp (single-run mode only)",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--world-size", type=int, default=1, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--plain-iters", type=int, default=30)
    parser.add_argument("--plain-warmup", type=int, default=5)
    parser.add_argument("--mpc-iters", type=int, default=10)
    parser.add_argument("--mpc-warmup", type=int, default=3)

    # Sweep options
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Sweep over models, world_sizes, and batch_sizes.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="cnn,mlp",
        help="Comma-separated list of models for sweep, e.g., 'cnn,mlp' or 'cnn'.",
    )
    parser.add_argument(
        "--world-sizes",
        type=str,
        default="1",
        help="Comma-separated list of world sizes for sweep, e.g., '1,2,3'.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,2,4,8,16",
        help="Comma-separated list of batch sizes for sweep, e.g., '1,4,16,64'.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of repetitions for each (model, world_size, batch_size) in sweep mode.",
    )

    args = parser.parse_args()

    if args.sweep:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        world_sizes = parse_int_list(args.world_sizes)
        batch_sizes = parse_int_list(args.batch_sizes)

        run_sweep(
            models=models,
            world_sizes=world_sizes,
            batch_sizes=batch_sizes,
            repeats=args.repeats,
            plain_iters=args.plain_iters,
            plain_warmup=args.plain_warmup,
            mpc_iters=args.mpc_iters,
            mpc_warmup=args.mpc_warmup,
        )
    else:
        # Single-run mode
        model_type = args.model.lower()
        mname = model_name_for_log(model_type)

        plain_latency = benchmark_plain_inference(
            model_type=model_type,
            batch_size=args.batch_size,
            iters=args.plain_iters,
            warmup=args.plain_warmup,
        )
        print(
            f"[Plain]      {mname}, batch={args.batch_size:3d}, device=cpu "
            f"→ {plain_latency*1000:.2f} ms / batch",
            flush=True,
        )

        if args.world_size == 1:
            mpc_latency = benchmark_crypten_singleparty(
                model_type=model_type,
                batch_size=args.batch_size,
                iters=args.mpc_iters,
                warmup=args.mpc_warmup,
            )
            if mpc_latency == mpc_latency:  # not NaN
                print(
                    f"[CrypTen-1P] {mname}, batch={args.batch_size:3d}, device=cpu "
                    f"→ {mpc_latency*1000:.2f} ms / batch",
                    flush=True,
                )
            else:
                print(f"[CrypTen-1P] {mname} → measurement failed.", flush=True)
        else:
            mpc_latency = benchmark_crypten_multiparty(
                model_type=model_type,
                world_size=args.world_size,
                batch_size=args.batch_size,
                iters=args.mpc_iters,
                warmup=args.mpc_warmup,
            )
            print(
                f"[CrypTen-{args.world_size}P] {mname}, batch={args.batch_size:3d}, device=cpu "
                f"→ {mpc_latency*1000:.2f} ms / batch",
                flush=True,
            )


if __name__ == "__main__":
    main()
