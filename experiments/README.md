# Experiments

## Overview

This folder groups runnable experiment scripts. Each subfolder has its own README with more detail.

## Quick Start (from repo root)

```
# Federated ensemble training + evaluation
python experiments/ensemble/federated_ensemble.py --dataset cifar10 --model resnet18 \
  --num-clients 5 --alpha 0.5 --epochs 5 --batch-size 128 --lr 0.001

# Reward allocation / fairness analysis
python experiments/ensemble/federated_ensemble_reward.py --dataset cifar10 --model resnet18 \
  --num-clients 5 --alpha 0.5 --epochs 5 --batch-size 128 --lr 0.001

# Single-host CrypTen benchmark
python experiments/SMPC-single_host/benchmark_crypten_parties_cpu.py \
  --model cnn --batch-size 32 --world-size 3

# Multi-host CrypTen benchmark (mp launcher)
python experiments/SMPC-multi-host/benchmark_crypten_multi_host_cpu.py --model lenet \
  --batch-size 32 --world-size 3 --launcher mp --runs 5

# Docker-based single-host benchmark
cd experiments/SMPC-single_host
docker compose up --build

# Docker: federated ensemble quick run
cd ../../experiments/ensemble
docker compose up --build
```
