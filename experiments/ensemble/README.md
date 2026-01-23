# Federated Ensemble Experiments

## Overview

Scripts for federated-style client training and ensemble inference on vision datasets.

## Run (Local Python)

```
# CIFAR-10
python federated_ensemble.py --dataset cifar10 --model resnet18 \
  --num-clients 5 --alpha 0.5 --epochs 5 --batch-size 128 --lr 0.001

# Reward allocation / fairness analysis
python federated_ensemble_reward.py --dataset cifar10 --model resnet18 \
  --num-clients 5 --alpha 0.5 --epochs 5 --batch-size 128 --lr 0.001
```

## Eval-only (Reuse Saved Checkpoints)

```
python federated_ensemble.py --dataset cifar10 --model resnet18 \
  --num-clients 5 --alpha 0.5 --batch-size 128 --eval-only
```

## Notes

- Use `--model-dir` to control where checkpoints are saved/loaded.
- For MedMNIST, pass `--dataset medmnist --medmnist-subset <name>`.

## Docker (Repeatable Run)

```
# From this folder
mkdir -p docker-output/models
docker compose up --build
```

Logs and checkpoints are written to ./docker-output. Default config runs on CPU (see docker-compose.yml).

## Customize via Environment Variables

```
DATASET=cifar10 MODEL=lenet NUM_CLIENTS=5 EPOCHS=2 \
 BATCH_SIZE=64 LR=0.0005 ALPHA=0.3 \
 docker compose up --build
```

## Extra CLI flags

```
EXTRA_ARGS="--eval-only --log-interval 10" docker compose up --build
```

## Reward Analysis

```
SCRIPT=federated_ensemble_reward.py \
DATASET=cifar10 MODEL=lenet NUM_CLIENTS=5 EPOCHS=2 \
BATCH_SIZE=64 LR=0.0005 ALPHA=0.3 \
docker compose up --build
```
