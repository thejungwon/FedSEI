#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Benchmark plain PyTorch vs CrypTen inference latency on CPU
for models: mlp / cnn / lenet / resnet18
with different number of parties (world_size).

- launcher=mp : run_multiprocess runs in one process; --runs gives mean±std
- launcher=env: run each party as a separate process (env vars required)
                in this mode, rank0 writes latency to a file via --run-id/--log-dir

Usage (mp):
  python benchmark_crypten_multi_host_cpu.py --model lenet --batch-size 32 --world-size 3 --launcher mp --runs 5

Usage (env):
  # Set env vars in each terminal/host (WORLD_SIZE/RANK/MASTER_ADDR/MASTER_PORT/RENDEZVOUS/...)
  python benchmark_crypten_multi_host_cpu.py --model lenet --batch-size 32 --world-size 3 --launcher env --run-id 0 --log-dir ./logs
"""

import argparse
import time
import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import crypten
import crypten.nn.onnx_converter as onnx_conv

onnx_conv._OPSET_VERSION = 11

import crypten.mpc as mpc
import multiprocessing as mp

# mac nested function + run_multiprocess
try:
    mp.set_start_method("fork", force=True)
except RuntimeError:
    pass

torch.set_num_threads(1)


# =========================
# 1) Models
# =========================


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SmallMLP(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class LeNet(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool = nn.AvgPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # (B, 6, 28, 28)
        x = self.pool(x)  # (B, 6, 14, 14)
        x = F.relu(self.conv2(x))  # (B, 16, 10, 10)
        x = self.pool(x)  # (B, 16, 5, 5)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def _build_resnet18(num_classes: int = 10):
    # torchvision optional dependency
    from torchvision.models import resnet18

    m = resnet18(num_classes=num_classes)
    # CIFAR-like tweak
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m


class ResNet18Wrapper(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.model = _build_resnet18(num_classes=num_classes)

    def forward(self, x):
        return self.model(x)


def create_model(model_type: str, num_classes: int = 10) -> nn.Module:
    t = model_type.lower()
    if t == "cnn":
        return SmallCNN(num_classes=num_classes)
    if t == "mlp":
        return SmallMLP(num_classes=num_classes)
    if t == "lenet":
        return LeNet(num_classes=num_classes)
    if t == "resnet18":
        return ResNet18Wrapper(num_classes=num_classes)
    raise ValueError(f"Unknown model_type: {model_type}")


# =========================
# 2) Utils
# =========================


def build_dummy_input(batch_size: int) -> torch.Tensor:
    return torch.randn(batch_size, 3, 32, 32)


def summarize_runs(latencies):
    xs = [x for x in latencies if x == x]  # drop NaN
    if len(xs) == 0:
        return float("nan"), float("nan"), 0
    mean = sum(xs) / len(xs)
    var = sum((x - mean) ** 2 for x in xs) / (len(xs) - 1) if len(xs) > 1 else 0.0
    std = math.sqrt(var)
    return mean, std, len(xs)


# =========================
# 3) Plain benchmark
# =========================


def benchmark_plain_once(
    model_type: str, batch_size: int, iters: int, warmup: int
) -> float:
    torch.set_num_threads(1)
    device = torch.device("cpu")
    model = create_model(model_type, num_classes=10).to(device)
    model.eval()
    x = build_dummy_input(batch_size).to(device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)

    start = time.time()
    with torch.no_grad():
        for _ in range(iters):
            _ = model(x)
    end = time.time()
    return (end - start) / iters


def benchmark_plain(
    model_type: str, batch_size: int, iters: int, warmup: int, runs: int
):
    lat = [
        benchmark_plain_once(model_type, batch_size, iters, warmup) for _ in range(runs)
    ]
    return summarize_runs(lat)


# =========================
# 4) CrypTen 1P
# =========================


def benchmark_crypten_1p_once(
    model_type: str, batch_size: int, iters: int, warmup: int
) -> float:
    torch.set_num_threads(1)
    crypten.init()
    device = torch.device("cpu")
    model = create_model(model_type, num_classes=10).to(device)
    model.eval()
    x = build_dummy_input(batch_size).to(device)

    try:
        cmodel = crypten.nn.from_pytorch(model, x)
    except ValueError as e:
        print(f"[CrypTen-1P] ({model_type}) from_pytorch failed: {e}")
        return float("nan")

    cmodel.encrypt()
    x_enc = crypten.cryptensor(x)

    for _ in range(warmup):
        _ = cmodel(x_enc)

    start = time.time()
    for _ in range(iters):
        _ = cmodel(x_enc)
    end = time.time()
    return (end - start) / iters


def benchmark_crypten_1p(
    model_type: str, batch_size: int, iters: int, warmup: int, runs: int
):
    lat = [
        benchmark_crypten_1p_once(model_type, batch_size, iters, warmup)
        for _ in range(runs)
    ]
    return summarize_runs(lat)


# =========================
# 5-A) CrypTen multiparty (mp launcher)
# =========================


def benchmark_crypten_mp_once(
    model_type: str, world_size: int, batch_size: int, iters: int, warmup: int
):
    """
    Returns latency (seconds/batch) from rank0, or NaN if unsupported.
    """
    assert world_size >= 2

    # shared object across subprocesses
    mgr = mp.Manager()
    out = mgr.dict()
    out["latency"] = float("nan")
    out["ok"] = True

    @mpc.run_multiprocess(world_size=world_size)
    def _worker():
        crypten.init()
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except AttributeError:
            pass

        comm = crypten.comm.get()
        rank = comm.get_rank()
        ws = comm.get_world_size()

        device = torch.device("cpu")
        model = create_model(model_type, num_classes=10).to(device)
        model.eval()
        x = build_dummy_input(batch_size).to(device)

        try:
            cmodel = crypten.nn.from_pytorch(model, x)
        except ValueError as e:
            if rank == 0:
                print(f"[CrypTen-{ws}P/mp] ({model_type}) from_pytorch failed: {e}")
                out["ok"] = False
            comm.barrier()
            return

        cmodel.encrypt()
        x_enc = crypten.cryptensor(x)

        for _ in range(warmup):
            _ = cmodel(x_enc)
        comm.barrier()

        if rank == 0:
            start = time.time()
        for _ in range(iters):
            _ = cmodel(x_enc)
        comm.barrier()
        if rank == 0:
            end = time.time()
            out["latency"] = (end - start) / iters

    _worker()
    return float(out["latency"]) if out.get("ok", True) else float("nan")


def benchmark_crypten_mp(
    model_type: str,
    world_size: int,
    batch_size: int,
    iters: int,
    warmup: int,
    runs: int,
):
    lat = [
        benchmark_crypten_mp_once(model_type, world_size, batch_size, iters, warmup)
        for _ in range(runs)
    ]
    return summarize_runs(lat)


# =========================
# 5-B) CrypTen multiparty (env launcher)
# =========================


def forward_ensemble(cmodel, x_enc, k: int):
    out = None
    for _ in range(k):
        y = cmodel(x_enc)
        out = y if out is None else (out + y)
    if k > 1:
        out = out / k
    return out


def benchmark_crypten_env(
    model_type: str,
    world_size: int,
    batch_size: int,
    iters: int,
    warmup: int,
    run_id: int,
    log_dir: str,
    quiet: bool,
    ensemble_k: int = 1,  # added
):
    """
    env-based init: run this script once per party.
    rank0 writes latency to a file:
      {log_dir}/lat_{model}_bs{bs}_ws{ws}_run{run_id}.txt
    """
    import socket

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except AttributeError:
        pass

    rank_env = os.environ.get("RANK", "NA")
    ws_env = os.environ.get("WORLD_SIZE", "NA")
    master_addr = os.environ.get("MASTER_ADDR", "NA")
    master_port = os.environ.get("MASTER_PORT", "NA")
    rendezvous = os.environ.get("RENDEZVOUS", "NA")
    backend = os.environ.get("DISTRIBUTED_BACKEND", "NA")
    iface = os.environ.get("GLOO_SOCKET_IFNAME", "NA")

    if not quiet:
        try:
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
        except Exception:
            hostname = "unknown"
            local_ip = "unknown"

        print(
            f"[DEBUG][pre-init] RANK(env)={rank_env}, WORLD_SIZE(env)={ws_env}, "
            f"MASTER_ADDR={master_addr}, MASTER_PORT={master_port}, "
            f"RENDEZVOUS={rendezvous}, BACKEND={backend}, "
            f"GLOO_SOCKET_IFNAME={iface}, hostname={hostname}, ip={local_ip}",
            flush=True,
        )

    try:
        crypten.init()
    except Exception as e:
        print(f"[ERROR] RANK={rank_env} crypten.init() failed: {repr(e)}", flush=True)
        raise

    comm = crypten.comm.get()
    rank = comm.get_rank()
    ws = comm.get_world_size()

    if ws != world_size and rank == 0 and not quiet:
        print(
            f"[Warning] args.world_size={world_size}, but env WORLD_SIZE={ws}. Using env WORLD_SIZE.",
            flush=True,
        )

    device = torch.device("cpu")
    model = create_model(model_type, num_classes=10).to(device)
    model.eval()
    x = build_dummy_input(batch_size).to(device)

    try:
        cmodel = crypten.nn.from_pytorch(model, x)
    except ValueError as e:
        if rank == 0:
            print(f"[CrypTen-{ws}P/env] ({model_type}) from_pytorch failed: {e}")
        comm.barrier()
        return

    cmodel.encrypt()
    x_enc = crypten.cryptensor(x)

    for _ in range(warmup):
        _ = forward_ensemble(cmodel, x_enc, ensemble_k)
    comm.barrier()

    if rank == 0:
        start = time.time()
    for _ in range(iters):
        _ = forward_ensemble(cmodel, x_enc, ensemble_k)
    comm.barrier()

    if rank == 0:
        end = time.time()
        latency = (end - start) / iters
        print(
            f"[CrypTen-{ws}P/env] {model_type}, K={ensemble_k}, batch={batch_size:3d}, device=cpu → "
            f"{latency*1000:.2f} ms / batch",
            flush=True,
        )

        os.makedirs(log_dir, exist_ok=True)
        out_path = os.path.join(
            log_dir,
            f"lat_{model_type}_K{ensemble_k}_bs{batch_size}_ws{ws}_run{run_id}.txt",
        )

        with open(out_path, "w") as f:
            f.write(f"{latency}\n")
        if not quiet:
            print(f"[DEBUG] wrote latency to {out_path}", flush=True)


# =========================
# 6) Main
# =========================


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model", type=str, default="cnn", choices=["mlp", "cnn", "lenet", "resnet18"]
    )
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--world-size", type=int, default=1, choices=[1, 2, 3, 4, 5])

    p.add_argument("--plain-iters", type=int, default=30)
    p.add_argument("--plain-warmup", type=int, default=5)

    p.add_argument("--mpc-iters", type=int, default=10)
    p.add_argument("--mpc-warmup", type=int, default=3)

    p.add_argument("--runs", type=int, default=5)

    p.add_argument("--launcher", type=str, default="mp", choices=["mp", "env"])

    # env-mode extras (only used when launcher=env)
    p.add_argument("--run-id", type=int, default=0)
    p.add_argument("--log-dir", type=str, default="./logs")
    p.add_argument(
        "--quiet", action="store_true", help="reduce debug prints (recommended)"
    )
    p.add_argument(
        "--ensemble-k", type=int, default=1, help="ensemble size K (1,2,4,...)"
    )

    args = p.parse_args()
    model_type = args.model.lower()

    # 1) Plain
    plain_mean, plain_std, n_plain = benchmark_plain(
        model_type=model_type,
        batch_size=args.batch_size,
        iters=args.plain_iters,
        warmup=args.plain_warmup,
        runs=args.runs,
    )
    print(
        f"[Plain]      {args.model}, batch={args.batch_size:3d}, cpu → "
        f"{plain_mean*1000:.2f} ± {plain_std*1000:.2f} ms / batch  (n={n_plain})"
    )

    # 2) CrypTen
    if args.world_size == 1:
        m_mean, m_std, n_m = benchmark_crypten_1p(
            model_type=model_type,
            batch_size=args.batch_size,
            iters=args.mpc_iters,
            warmup=args.mpc_warmup,
            runs=args.runs,
        )
        if m_mean == m_mean:
            print(
                f"[CrypTen-1P] {args.model}, batch={args.batch_size:3d}, cpu → "
                f"{m_mean*1000:.2f} ± {m_std*1000:.2f} ms / batch  (n={n_m})"
            )
        else:
            print(f"[CrypTen-1P] {args.model} → measurement failed (unsupported ops).")
        return

    # multiparty
    if args.launcher == "mp":
        m_mean, m_std, n_m = benchmark_crypten_mp(
            model_type=model_type,
            world_size=args.world_size,
            batch_size=args.batch_size,
            iters=args.mpc_iters,
            warmup=args.mpc_warmup,
            runs=args.runs,
        )
        if m_mean == m_mean:
            print(
                f"[CrypTen-{args.world_size}P/mp] {args.model}, batch={args.batch_size:3d}, cpu → "
                f"{m_mean*1000:.2f} ± {m_std*1000:.2f} ms / batch  (n={n_m})"
            )
        else:
            print(
                f"[CrypTen-{args.world_size}P/mp] {args.model} → measurement failed (unsupported ops)."
            )
    else:
        # env mode: single run per invocation, rank0 writes a file
        benchmark_crypten_env(
            model_type=model_type,
            world_size=args.world_size,
            batch_size=args.batch_size,
            iters=args.mpc_iters,
            warmup=args.mpc_warmup,
            run_id=args.run_id,
            log_dir=args.log_dir,
            quiet=args.quiet,
            ensemble_k=args.ensemble_k,
        )
        if int(os.environ.get("RANK", "0")) == 0:
            print(
                f"[env-note] rank0 wrote a log file in {args.log_dir}. "
                f"Repeat with --run-id 0..{args.runs-1} to get n={args.runs} runs."
            )


if __name__ == "__main__":
    main()
