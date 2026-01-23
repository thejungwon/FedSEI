import os
import json
import threading
import multiprocessing as mp
import traceback
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import crypten
import crypten.nn
import crypten.nn.onnx_converter as onnx_conv
from crypten.mpc import MPCTensor
from eth_account import Account
from eth_account.messages import encode_defunct
from hexbytes import HexBytes
from web3 import Web3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

RANK = int(os.environ["RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
MODEL_DIR = os.environ.get(
    "MODEL_DIR", os.path.join(os.path.dirname(__file__), "models")
)

# Fixed keys for PoC only (32-byte hex strings)
PARTY_PRIVATE_KEYS = {
    0: "0x1111111111111111111111111111111111111111111111111111111111111111",
    1: "0x2222222222222222222222222222222222222222222222222222222222222222",
    2: "0x3333333333333333333333333333333333333333333333333333333333333333",
}

jobs: Dict[str, Dict[str, Any]] = {}
jobs_lock = threading.Lock()
JOB_STORE_DIR = "/tmp/mpc_jobs"
os.makedirs(JOB_STORE_DIR, exist_ok=True)


class ShareRequest(BaseModel):
    job_id: str
    job_token: str
    share: str
    input_meta: Dict[str, Any]


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
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_dummy_model():
    # LeNet-style classifier; kept in PyTorch for future weight loading.
    return LeNet()


def _load_state_dict(path: str) -> Dict[str, Any]:
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        return state["state_dict"]
    return state


def _get_model_paths() -> list:
    if not os.path.isdir(MODEL_DIR):
        raise FileNotFoundError(f"MODEL_DIR not found: {MODEL_DIR}")
    paths = [
        os.path.join(MODEL_DIR, name)
        for name in os.listdir(MODEL_DIR)
        if name.endswith(".pt")
    ]
    paths.sort()
    if not paths:
        raise FileNotFoundError(f"No .pt model files found in {MODEL_DIR}")
    return paths


def _compute_signature(job_id: str, client_address: str) -> str:
    job_id_bytes = HexBytes(job_id)
    client_checksum = Web3.to_checksum_address(client_address)
    message_hash = Web3.solidity_keccak(
        ["bytes32", "address"], [job_id_bytes, client_checksum]
    )
    signed = Account.sign_message(
        encode_defunct(message_hash),
        private_key=PARTY_PRIVATE_KEYS[RANK],
    )
    return "0x" + signed.signature.hex()


def _job_path(job_id: str) -> str:
    return os.path.join(JOB_STORE_DIR, f"{job_id}.json")


def _write_job_file(job_id: str, payload: Dict[str, Any]):
    path = _job_path(job_id)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f)
    os.replace(tmp_path, path)


def _refresh_job(job_id: str):
    path = _job_path(job_id)
    if not os.path.exists(path):
        return
    with open(path, "r") as f:
        payload = json.load(f)
    with jobs_lock:
        if job_id not in jobs:
            return
        jobs[job_id].update(payload)


def _run_mpc_process(job_id: str, share: str, input_meta: Dict[str, Any]):
    print(f"[party{RANK}] job {job_id}: MPC process start", flush=True)
    try:
        print(f"[party{RANK}] job {job_id}: initializing CrypTen...", flush=True)
        crypten.init()
        print(f"[party{RANK}] job {job_id}: CrypTen initialized", flush=True)
        print(f"[party{RANK}] job {job_id}: decoding share", flush=True)
        share_tensor = torch.tensor(json.loads(share), dtype=torch.long)
        precision_bits = int(input_meta.get("precision_bits", 16))

        # Wait for all parties to receive their shares.
        print(f"[party{RANK}] job {job_id}: waiting at barrier", flush=True)
        dist.barrier()
        print(f"[party{RANK}] job {job_id}: barrier passed", flush=True)

        # Build MPCTensor directly from local share (no plaintext reconstruction).
        x = MPCTensor.from_shares(share_tensor, precision=precision_bits)
        print(f"[party{RANK}] job {job_id}: MPCTensor from shares ready", flush=True)

        onnx_conv._OPSET_VERSION = 11
        output = None
        model_paths = _get_model_paths()
        print(
            f"[party{RANK}] job {job_id}: starting ensemble (K={len(model_paths)})",
            flush=True,
        )
        for idx, model_path in enumerate(model_paths):
            model = get_dummy_model()
            model.load_state_dict(_load_state_dict(model_path))
            model.eval()
            cmodel = crypten.nn.from_pytorch(
                model, dummy_input=torch.randn(1, 3, 32, 32)
            )
            # Log a lightweight fingerprint to confirm weights loaded.
            with torch.no_grad():
                w = model.conv1.weight
                print(
                    f"[party{RANK}] job {job_id}: model{idx} weight mean {w.mean().item():.6f}, std {w.std().item():.6f}",
                    flush=True,
                )
            for module in cmodel.modules():
                if hasattr(module, "alpha") and isinstance(module.alpha, float):
                    module.alpha = int(module.alpha)
                if hasattr(module, "beta") and isinstance(module.beta, float):
                    module.beta = int(module.beta)
            print(
                f"[party{RANK}] job {job_id}: model{idx} wrapped ({os.path.basename(model_path)})",
                flush=True,
            )
            cmodel.encrypt()
            print(f"[party{RANK}] job {job_id}: model{idx} encrypted", flush=True)

            print(f"[party{RANK}] job {job_id}: model{idx} forward pass", flush=True)
            out_i = cmodel(x)
            output = out_i if output is None else (output + out_i)
        output = output / len(model_paths)
        print(f"[party{RANK}] job {job_id}: forward pass done", flush=True)

        # Store local share of the result; client will reconstruct.
        result_share = output._tensor.share
        precision_bits = output._tensor.encoder._precision_bits
        print(f"[party{RANK}] job {job_id}: result share ready", flush=True)

        client_addr = input_meta["client_address"]
        print(f"[party{RANK}] job {job_id}: signing completion", flush=True)
        signature = _compute_signature(job_id, client_addr)
        _write_job_file(
            job_id,
            {
                "status": "DONE",
                "result_share": result_share.tolist(),
                "precision_bits": precision_bits,
                "signature": signature,
                "error": None,
            },
        )
        print(f"[party{RANK}] job {job_id}: DONE", flush=True)
    except Exception as exc:
        tb = traceback.format_exc()
        _write_job_file(
            job_id,
            {
                "status": "FAILED",
                "result_share": None,
                "precision_bits": None,
                "signature": None,
                "error": f"{exc}\n{tb}",
            },
        )
        print(f"[party{RANK}] job {job_id}: FAILED {exc}", flush=True)
        print(tb, flush=True)


@app.on_event("startup")
async def startup_event():
    # Defer full CrypTen init until the first share arrives to avoid startup hangs.
    pass


@app.post("/share")
async def receive_share(request: ShareRequest):
    with jobs_lock:
        if request.job_id in jobs:
            raise HTTPException(status_code=400, detail="Job already exists")
        jobs[request.job_id] = {
            "share": request.share,
            "input_meta": request.input_meta,
            "job_token": request.job_token,
            "status": "RUNNING",
            "result_share": None,
            "precision_bits": None,
            "signature": None,
            "error": None,
        }

    proc = mp.Process(
        target=_run_mpc_process,
        args=(request.job_id, request.share, request.input_meta),
        daemon=True,
    )
    proc.start()
    return {"status": "accepted"}


@app.get("/status")
async def get_status(job_id: str):
    _refresh_job(job_id)
    with jobs_lock:
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        return {"status": jobs[job_id]["status"]}


@app.get("/result_share")
async def get_result_share(job_id: str):
    _refresh_job(job_id)
    with jobs_lock:
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        if jobs[job_id]["status"] != "DONE":
            raise HTTPException(status_code=400, detail="Job not done")
        return {
            "result_share": jobs[job_id]["result_share"],
            "precision_bits": jobs[job_id]["precision_bits"],
        }


@app.get("/completion_signature")
async def get_completion_signature(job_id: str):
    _refresh_job(job_id)
    with jobs_lock:
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        if jobs[job_id]["status"] != "DONE":
            raise HTTPException(status_code=400, detail="Job not done")
        return {"signature": jobs[job_id]["signature"]}
