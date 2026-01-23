import json
import os
import time
import uuid

import requests
import torch
from torchvision import datasets, transforms
from eth_account import Account
from hexbytes import HexBytes
from solcx import compile_standard, install_solc, set_solc_version
try:
    from solcx import set_solc_binary
except ImportError:
    set_solc_binary = None
from web3 import Web3
from crypten.encoder import FixedPointEncoder


def log(msg: str):
    print(f"[client] {msg}", flush=True)


anvil_url = os.environ.get("ANVIL_URL", "http://localhost:8545")
party_urls_env = os.environ.get("PARTY_URLS")
if party_urls_env:
    party_urls = [u.strip() for u in party_urls_env.split(",") if u.strip()]
else:
    party_urls = [
        "http://localhost:8000",
        "http://localhost:8001",
        "http://localhost:8002",
    ]

# Connect to local Anvil
log(f"connecting to local Anvil at {anvil_url}")
try:
    rpc_probe = requests.post(
        anvil_url,
        json={"jsonrpc": "2.0", "method": "eth_blockNumber", "params": [], "id": 1},
        timeout=3,
    )
    log(f"rpc probe status: {rpc_probe.status_code}")
    log(f"rpc probe body: {rpc_probe.text[:200]}")
except Exception as exc:
    log(f"rpc probe error: {exc}")
w3 = Web3(Web3.HTTPProvider(anvil_url, request_kwargs={"timeout": 3}))
for attempt in range(1, 31):
    try:
        if w3.is_connected():
            log("connected to Anvil")
            break
        log(f"anvil not ready yet (attempt {attempt}/30)")
    except Exception as exc:
        log(f"anvil connection error on attempt {attempt}/30: {exc}")
    time.sleep(1)
else:
    raise RuntimeError("failed to connect to Anvil after 30 attempts")

client_address = w3.eth.accounts[0]
w3.eth.default_account = client_address
log(f"using client address: {client_address}")

def load_compiled_artifact(path: str):
    with open(path, "r") as f:
        data = json.load(f)
    if "abi" in data and "bytecode" in data:
        return data["abi"], data["bytecode"]
    if "contracts" in data:
        contracts = data["contracts"]
        if "Escrow.sol" in contracts and "Escrow" in contracts["Escrow.sol"]:
            iface = contracts["Escrow.sol"]["Escrow"]
            return iface["abi"], iface["evm"]["bytecode"]["object"]
        # solc --combined-json output
        for key, value in contracts.items():
            if key.endswith(":Escrow") and "abi" in value and "bin" in value:
                abi_raw = value["abi"]
                if isinstance(abi_raw, str):
                    abi = json.loads(abi_raw)
                else:
                    abi = abi_raw
                return abi, value["bin"]
    raise ValueError(f"Unsupported artifact format: {path}")


# Compile contract (or load precompiled artifact)
artifact_path = os.environ.get("ESCROW_ARTIFACT")
if artifact_path:
    log(f"loading Escrow artifact from {artifact_path}")
    abi, bytecode = load_compiled_artifact(artifact_path)
else:
    log("compiling Escrow contract")
    solc_binary = os.environ.get("SOLC_BINARY")
    if solc_binary and set_solc_binary is not None:
        log(f"using SOLC_BINARY={solc_binary}")
        set_solc_binary(solc_binary)
    elif solc_binary:
        log("SOLC_BINARY is set but solcx does not support set_solc_binary.")
    try:
        set_solc_version("0.8.20")
    except Exception as exc:
        log(f"solc 0.8.20 not found, installing: {exc}")
        install_solc("0.8.20")
        set_solc_version("0.8.20")
    with open("../contract/Escrow.sol", "r") as f:
        contract_source = f.read()

    compiled_sol = compile_standard(
        {
            "language": "Solidity",
            "sources": {"Escrow.sol": {"content": contract_source}},
            "settings": {
                "outputSelection": {
                    "*": {"*": ["abi", "evm.bytecode"]},
                }
            },
        }
    )
    contract_interface = compiled_sol["contracts"]["Escrow.sol"]["Escrow"]
    abi = contract_interface["abi"]
    bytecode = contract_interface["evm"]["bytecode"]["object"]

# Deploy contract
log("deploying Escrow contract")
Escrow = w3.eth.contract(abi=abi, bytecode=bytecode)
tx_hash = Escrow.constructor().transact()
tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
contract = w3.eth.contract(address=tx_receipt.contractAddress, abi=abi)
log(f"Escrow deployed at {tx_receipt.contractAddress}")

# Fixed party private keys (PoC only)
party_private_keys = [
    "0x1111111111111111111111111111111111111111111111111111111111111111",
    "0x2222222222222222222222222222222222222222222222222222222222222222",
    "0x3333333333333333333333333333333333333333333333333333333333333333",
]
party_accounts = [Account.from_key(pk) for pk in party_private_keys]
party_addresses = [acct.address for acct in party_accounts]

# Party endpoints
log(f"party endpoints: {party_urls}")

# Generate job identifiers
log("generating job identifiers")
job_name = f"job-{uuid.uuid4().hex}"
job_id = Web3.keccak(text=job_name).hex()
job_token = uuid.uuid4().hex
log(f"job_id: {job_id}")

# Input tensor (CIFAR-10 test sample)
log("loading CIFAR-10 test sample")
data_root = os.environ.get("CIFAR10_ROOT", "./data")
sample_index = int(os.environ.get("CIFAR10_INDEX", "1"))
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616),
        ),
    ]
)
test_set = datasets.CIFAR10(
    root=data_root, train=False, download=True, transform=transform
)
image, label = test_set[sample_index]
input_tensor = image.unsqueeze(0)
log(f"input_tensor shape: {list(input_tensor.shape)}, label: {label}")

# Simple additive secret sharing for PoC
log("creating additive shares")
share0 = torch.randn_like(input_tensor)
share1 = torch.randn_like(input_tensor)
share2 = input_tensor - share0 - share1
shares = [share0, share1, share2]

precision_bits = 16
encoder = FixedPointEncoder(precision_bits=precision_bits)
encoded_shares = [encoder.encode(s) for s in shares]
share_strs = [json.dumps(s.tolist()) for s in encoded_shares]

# Create job on contract
log("creating job on escrow contract")
deposit = w3.to_wei(1, "ether")
job_id_bytes = HexBytes(job_id)
tx = contract.functions.createJob(job_id_bytes, party_addresses).transact(
    {"value": deposit}
)
w3.eth.wait_for_transaction_receipt(tx)
log("escrow job created and funded")

# Send shares
log("sending shares to parties")
input_meta = {
    "client_address": client_address,
    "input_shape": list(input_tensor.shape),
    "job_id": job_id,
    "precision_bits": precision_bits,
}
for i in range(3):
    data = {
        "job_id": job_id,
        "job_token": job_token,
        "share": share_strs[i],
        "input_meta": input_meta,
    }
    log(f"POST /share -> party{i} ({party_urls[i]})")
    resp = requests.post(f"{party_urls[i]}/share", json=data, timeout=10)
    resp.raise_for_status()
    log(f"party{i} accepted share: {resp.status_code}")

# Poll for status on party0
log("polling party0 for status")
while True:
    resp = requests.get(f"{party_urls[0]}/status?job_id={job_id}", timeout=10)
    resp.raise_for_status()
    status = resp.json()["status"]
    log(f"status: {status}")
    if status == "DONE":
        break
    time.sleep(1)

# Get result shares and reconstruct locally
log("fetching result shares from all parties")
result_shares = []
precision_bits = None
while True:
    try:
        result_shares.clear()
        precision_bits = None
        for i in range(3):
            resp = requests.get(
                f"{party_urls[i]}/result_share?job_id={job_id}", timeout=10
            )
            resp.raise_for_status()
            payload = resp.json()
            if precision_bits is None:
                precision_bits = payload["precision_bits"]
            result_shares.append(payload["result_share"])
        break
    except requests.HTTPError:
        time.sleep(1)

encoder = FixedPointEncoder(precision_bits=precision_bits)
share_tensors = [torch.tensor(s, dtype=torch.long) for s in result_shares]
sum_shares = sum(share_tensors)
decoded = encoder.decode(sum_shares)
result = decoded.tolist()
pred = int(decoded.argmax().item())
log(f"result: {result}")
log(f"prediction: {pred}, label: {label}")

# Collect signatures
log("collecting completion signatures")
sigs = []
for i in range(3):
    resp = requests.get(
        f"{party_urls[i]}/completion_signature?job_id={job_id}", timeout=10
    )
    resp.raise_for_status()
    sig_hex = resp.json()["signature"]
    log(f"party{i} signature raw: {sig_hex}")
    if sig_hex.startswith("0x0x"):
        sig_hex = "0x" + sig_hex[4:]
    sigs.append(HexBytes(sig_hex))
    log(f"party{i} signature received")

# Complete job
log("calling completeJob on escrow")
tx = contract.functions.completeJob(job_id_bytes, sigs[0], sigs[1], sigs[2]).transact()
w3.eth.wait_for_transaction_receipt(tx)
log("job completed and escrow paid out")
