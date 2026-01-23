# FedINF-PoC

## Overview

This repository contains:
- `PoC/`: MPC-as-a-Service proof-of-concept (client + parties + contract)
- `experiments/`: research scripts for federated ensemble and SMPC benchmarks

## Structure

- `PoC/` for end-to-end PoC
- `experiments/` for standalone research scripts

## Quick Start (no multi-host)

## PoC (Docker + client)

```
# start Anvil + 3 parties
cd PoC
docker-compose up --build
```

```
# in another terminal (choose one)
cd PoC
docker-compose run --rm --use-aliases client

# or local Python
cd PoC/client
pip install -r requirements.txt
python client.py
```

```
# Precompile and run with ESCROW_ARTIFACT (recommended on Apple Silicon)
cd PoC
docker run --rm -v "$PWD/contract:/src" ethereum/solc:0.8.20 \
  --combined-json abi,bin /src/Escrow.sol > ./contract/Escrow.compiled.json
docker-compose run --rm --use-aliases \
  -e ESCROW_ARTIFACT=/app/PoC/contract/Escrow.compiled.json client
```

## Experiments (local Python)

```
# Federated ensemble training + evaluation
python experiments/ensemble/federated_ensemble.py --dataset cifar10 --model resnet18 \
  --num-clients 5 --alpha 0.5 --epochs 5 --batch-size 128 --lr 0.001

# Reward allocation / fairness analysis
python experiments/ensemble/federated_ensemble_reward.py --dataset cifar10 --model resnet18 \
  --num-clients 5 --alpha 0.5 --epochs 5 --batch-size 128 --lr 0.001

# SMPC single-host benchmark (parties)
python experiments/SMPC-single_host/benchmark_crypten_parties_cpu.py \
  --model cnn --batch-size 32 --world-size 3

# SMPC single-host benchmark (CNN, single-process)
python experiments/SMPC-single_host/benchmark_crypten_cnn.py \
  --model small --batch-size 32 --device cpu

# SMPC single-host benchmark (MLP, single-process)
python experiments/SMPC-single_host/benchmark_crypten_mlp.py \
  --model small --batch-size 32 --device cpu
```

## Experiments (Docker)

```
# Federated ensemble (configurable via env vars)
cd experiments/ensemble
docker compose up --build

# SMPC single-host
cd ../../experiments/SMPC-single_host
docker compose up --build
```

## More details

- `PoC/README.md`
- `experiments/README.md`
