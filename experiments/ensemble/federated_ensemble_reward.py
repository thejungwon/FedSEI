#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Federated-style independent client training on CIFAR-10/CIFAR-100 using a Dirichlet Non-IID split,
then ensemble inference + reward allocation / fairness analysis.

포커스:
  - Collaboration: hard / soft / entropy-weighted ensemble accuracy
  - Incentive: label-free reward mechanisms vs ex-post merit fairness (using test accuracy)

Reward schemes (label-free at inference time):
  1) Equal split (1/n)
  2) Final-agreement split (only models that agree with final ensemble prediction share the reward)
  3) Entropy-based split (more confident models get larger share per sample)

Fairness (ex-post, for analysis only):
  - "Merit" = per-client test accuracy (can be changed later)
  - Skill-based fairness F_skill in [0, 1]:
      * 이상적 분배 p_i ∝ acc_i
      * 실제 분배 r_i ∝ reward_i
      * L1(p, r)를 기반으로 1 - D/2 로 변환 (0=최악, 1=완전 score-proportional)

Usage (example):
    # 1) Train clients and save models
    python federated_ensemble_reward.py --dataset cifar10 --model resnet18 \
        --num-clients 5 --alpha 0.5 --epochs 5 --batch-size 128 --lr 0.001

    # 2) Later, eval-only with the same hyperparams (no retraining)
    python federated_ensemble_reward.py --dataset cifar10 --model resnet18 \
        --num-clients 5 --alpha 0.5 --batch-size 128 --eval-only
"""

import argparse
import os
import random
from copy import deepcopy
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

# wandb (optional)
try:
    import wandb
except ImportError:
    wandb = None

try:
    import medmnist
    from medmnist import INFO as MEDMNIST_INFO
except ImportError:
    medmnist = None
    MEDMNIST_INFO = None
# =============================
# Utils
# =============================


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def available_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_device(arg_device: str):
    if arg_device == "auto":
        return available_device()
    if arg_device == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            print("[Warning] MPS not available. Falling back to CPU.", flush=True)
            return torch.device("cpu")
    # e.g. "cuda:0", "cpu"
    return torch.device(arg_device)


def get_client_ckpt_path(args, cid: int) -> str:
    """Make a unique checkpoint path for a given client & hyperparams."""
    fname = (
        f"{args.dataset}"
        + (f"-{args.medmnist_subset}" if args.dataset == "medmnist" else "")
        + f"_{args.model}_C{args.num_clients}_"
        f"a{args.alpha}_seed{args.seed}_client{cid}.pt"
    )
    return os.path.join(args.model_dir, fname)


# =============================
# Models
# =============================


class LeNet(nn.Module):
    """Classic LeNet-5 style network adapted for 32x32x3 input."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)  # 32 -> 28
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # 14 -> 10
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [B, 6, 14, 14]
        x = self.pool(F.relu(self.conv2(x)))  # [B, 16, 5, 5]
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 64

        # CIFAR: 3x3 conv, no maxpool
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)  # 32x32
        out = self.layer2(out)  # 16x16
        out = self.layer3(out)  # 8x8
        out = self.layer4(out)  # 4x4
        out = self.avgpool(out)  # 1x1
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def create_model(model_name: str, num_classes: int) -> nn.Module:
    model_name = model_name.lower()
    if model_name == "lenet":
        return LeNet(num_classes=num_classes)
    elif model_name == "resnet18":
        return ResNet18(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")


# =============================
# Dirichlet Non-IID splitter
# =============================


def dirichlet_noniid_split(
    labels: np.ndarray, num_clients: int, alpha: float
) -> List[np.ndarray]:
    """Split indices into Non-IID partitions using a class-wise Dirichlet."""
    num_classes = int(labels.max()) + 1
    idxs_by_class = [np.where(labels == c)[0] for c in range(num_classes)]
    for idxs in idxs_by_class:
        np.random.shuffle(idxs)

    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        idxs_c = idxs_by_class[c]
        if len(idxs_c) == 0:
            continue
        proportions = np.random.dirichlet(alpha=[alpha] * num_clients)
        splits = (proportions * len(idxs_c)).astype(int)

        # rounding correction
        while splits.sum() < len(idxs_c):
            splits[np.argmin(splits)] += 1
        while splits.sum() > len(idxs_c):
            splits[np.argmax(splits)] -= 1

        start = 0
        for i, count in enumerate(splits):
            if count > 0:
                client_indices[i].extend(idxs_c[start : start + count].tolist())
            start += count

    for i in range(num_clients):
        random.shuffle(client_indices[i])
        client_indices[i] = np.array(client_indices[i], dtype=np.int64)

    return client_indices


# =============================
# Data loading
# =============================


def get_dataloaders(
    data_dir: str,
    batch_size: int,
    dataset: str,
    num_workers: int = 2,
    medmnist_subset: str = None,
):
    """
    dataset: "cifar10", "cifar100", "emnist", "fashionmnist", "medmnist"
    """
    dataset = dataset.lower()

    # ===== CIFAR family =====
    if dataset in ["cifar10", "cifar100"]:
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2470, 0.2435, 0.2616),
                ),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2470, 0.2435, 0.2616),
                ),
            ]
        )

        if dataset == "cifar10":
            trainset = datasets.CIFAR10(
                root=data_dir, train=True, download=True, transform=transform_train
            )
            testset = datasets.CIFAR10(
                root=data_dir, train=False, download=True, transform=transform_test
            )
            num_classes = 10
            in_channels = 3
        else:  # cifar100
            trainset = datasets.CIFAR100(
                root=data_dir, train=True, download=True, transform=transform_train
            )
            testset = datasets.CIFAR100(
                root=data_dir, train=False, download=True, transform=transform_test
            )
            num_classes = 100
            in_channels = 3

    # ===== MNIST family (EMNIST / FashionMNIST) =====
    elif dataset in ["emnist", "fashionmnist"]:
        # Expand 1x28x28 to 3x32x32
        transform_train = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),  # [1, 32, 32]
                transforms.Lambda(lambda x: x.expand(3, -1, -1)),  # [3, 32, 32]
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.expand(3, -1, -1)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        if dataset == "fashionmnist":
            trainset = datasets.FashionMNIST(
                root=data_dir, train=True, download=True, transform=transform_train
            )
            testset = datasets.FashionMNIST(
                root=data_dir, train=False, download=True, transform=transform_test
            )
            num_classes = 10
        else:  # emnist
            # balanced split (47 classes)
            trainset = datasets.EMNIST(
                root=data_dir,
                split="balanced",
                train=True,
                download=True,
                transform=transform_train,
            )
            testset = datasets.EMNIST(
                root=data_dir,
                split="balanced",
                train=False,
                download=True,
                transform=transform_test,
            )
            num_classes = 47
        in_channels = 3  # expanded to 3 channels in transforms

    elif dataset == "medmnist":
        if medmnist is None or MEDMNIST_INFO is None:
            raise RuntimeError(
                "medmnist is not installed. Install with `pip install medmnist`."
            )
        if medmnist_subset is None:
            raise ValueError(
                "For dataset='medmnist', you must specify --medmnist-subset "
                "(e.g., --medmnist-subset pathmnist)."
            )

        subset = medmnist_subset.lower()
        if subset not in MEDMNIST_INFO:
            raise ValueError(
                f"Unknown MedMNIST subset: {subset}. "
                f"Available keys include: {list(MEDMNIST_INFO.keys())[:10]} ..."
            )

        info = MEDMNIST_INFO[subset]
        DataClass = getattr(medmnist, info["python_class"])
        channels = info["n_channels"]
        num_classes = len(info["label"])

        # Normalize to 3x32x32 (preserve previous logic)
        base_transforms_train = []
        base_transforms_train.append(transforms.Resize(32))
        base_transforms_train.append(transforms.RandomHorizontalFlip())
        if channels == 1:
            base_transforms_train.append(transforms.Grayscale(num_output_channels=3))
        base_transforms_train.append(transforms.ToTensor())
        base_transforms_train.append(
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
            )
        )
        transform_train = transforms.Compose(base_transforms_train)

        base_transforms_test = []
        base_transforms_test.append(transforms.Resize(32))
        if channels == 1:
            base_transforms_test.append(transforms.Grayscale(num_output_channels=3))
        base_transforms_test.append(transforms.ToTensor())
        base_transforms_test.append(
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
            )
        )
        transform_test = transforms.Compose(base_transforms_test)

        trainset = DataClass(
            root=data_dir,
            split="train",
            transform=transform_train,
            download=True,
        )
        testset = DataClass(
            root=data_dir,
            split="test",
            transform=transform_test,
            download=True,
        )

        test_loader = DataLoader(
            testset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        in_channels = 3  # aligned to 3 channels above

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    test_loader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return trainset, test_loader, num_classes, in_channels


def make_client_loaders(
    trainset, client_indices: List[np.ndarray], batch_size: int, num_workers: int = 2
):
    loaders = []
    for idxs in client_indices:
        subset = Subset(trainset, indices=idxs)
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=False,
        )
        loaders.append(loader)
    return loaders


# =============================
# Training & Evaluation
# =============================


def _normalize_targets_for_ce(targets: torch.Tensor) -> torch.Tensor:
    """
    Normalize labels for CrossEntropy:
    - If shape [B, 1], squeeze to [B]
    - If one-hot [B, C], take argmax
    - Else, return as-is
    """
    if targets.ndim > 1:
        if targets.size(-1) > 1:
            targets = targets.argmax(dim=-1)
        else:
            targets = targets.squeeze(-1)
    return targets.long()


def train_one_client(
    model: nn.Module,
    loader: DataLoader,
    device,
    epochs: int,
    lr: float,
    weight_decay: float = 5e-4,
    early_stop: bool = False,
    patience: int = 5,
    val_fraction: float = 0.1,
):
    """
    Train a single client model with optional validation-based early stopping.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Optional client-wise train/val split
    if early_stop and len(loader.dataset) > 1 and 0.0 < val_fraction < 1.0:
        dataset = loader.dataset
        n_total = len(dataset)
        n_val = max(1, int(n_total * val_fraction))
        n_train = n_total - n_val
        if n_train <= 0:
            train_loader = loader
            val_loader = None
            print(
                "  [EarlyStop] Not enough samples to create val set. Using all for training.",
                flush=True,
            )
        else:
            train_set, val_set = random_split(dataset, [n_train, n_val])
            train_loader = DataLoader(
                train_set,
                batch_size=loader.batch_size,
                shuffle=True,
                num_workers=loader.num_workers,
                drop_last=False,
            )
            val_loader = DataLoader(
                val_set,
                batch_size=loader.batch_size,
                shuffle=False,
                num_workers=loader.num_workers,
                drop_last=False,
            )
            print(
                f"  [EarlyStop] Train samples: {n_train}, Val samples: {n_val}",
                flush=True,
            )
    else:
        train_loader = loader
        val_loader = None

    best_state = None
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for ep in range(epochs):
        # Train
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            targets = _normalize_targets_for_ce(targets)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        avg_loss = running_loss / max(1, total)
        train_acc = 100.0 * correct / max(1, total)

        if val_loader is not None:
            model.eval()
            val_loss_sum = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    targets = _normalize_targets_for_ce(targets)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss_sum += loss.item() * inputs.size(0)
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()

            val_loss = val_loss_sum / max(1, val_total)
            val_acc = 100.0 * val_correct / max(1, val_total)

            print(
                f"  Epoch {ep+1:03d}: "
                f"train_loss={avg_loss:.4f} train_acc={train_acc:.2f}% | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}%",
                flush=True,
            )

            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_state = deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if early_stop and epochs_no_improve >= patience:
                    print(
                        f"  [EarlyStop] No improvement for {patience} epochs. Stopping early.",
                        flush=True,
                    )
                    break
        else:
            print(
                f"  Epoch {ep+1:03d}: loss={avg_loss:.4f} acc={train_acc:.2f}%",
                flush=True,
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def collect_logits(
    models: List[nn.Module], loader: DataLoader, device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return (logits, labels)
      logits shape: [num_models, N, num_classes]
      labels shape: [N]
    """
    for m in models:
        m.eval()
        m.to(device)

    all_logits = []
    all_labels = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = _normalize_targets_for_ce(targets)
            batch_logits = []
            for m in models:
                out = m(inputs)
                batch_logits.append(out.detach().cpu())
            all_logits.append(torch.stack(batch_logits, dim=0))
            all_labels.append(targets.clone())

    logits = torch.cat(all_logits, dim=1)  # [K, N, C]
    labels = torch.cat(all_labels, dim=0)  # [N]
    return logits, labels


def accuracy_from_preds(preds: torch.Tensor, labels: torch.Tensor) -> float:
    correct = (preds == labels).sum().item()
    return 100.0 * correct / labels.numel()


def acc_from_logits(
    logits_k_nc: torch.Tensor,
    labels: torch.Tensor,
    method: str = "soft",
    eps: float = 1e-12,
) -> float:
    """
    Ensembling over K predictors with logits shape [K, N, C].
    method in {"hard", "soft", "entropy"}
    """
    K, N, C = logits_k_nc.shape
    if method == "hard":
        preds = logits_k_nc.argmax(dim=-1)  # [K, N]
        final = []
        for j in range(N):
            votes = preds[:, j]
            binc = torch.bincount(votes, minlength=C)
            final.append(int(torch.argmax(binc)))
        final = torch.tensor(final, dtype=torch.long)
        return accuracy_from_preds(final, labels)

    probs = F.softmax(logits_k_nc, dim=-1)  # [K, N, C]
    if method == "soft":
        mean_prob = probs.mean(dim=0)  # [N, C]
        final = mean_prob.argmax(dim=-1)
        return accuracy_from_preds(final, labels)
    elif method == "entropy":
        ent = -(probs * (probs + eps).log()).sum(dim=-1)  # [K, N]
        w = torch.exp(-ent)  # low entropy -> high weight
        w = w / (w.sum(dim=0, keepdim=True) + eps)
        weighted = (w.unsqueeze(-1) * probs).sum(dim=0)  # [N, C]
        final = weighted.argmax(dim=-1)
        return accuracy_from_preds(final, labels)
    else:
        raise ValueError(f"Unknown method: {method}")


# =============================
# Reward mechanisms (label-free)
# =============================


def compute_reward_equal(num_clients: int) -> np.ndarray:
    """
    Equal split (1/n).
    Scale does not affect fairness, so normalize to sum to 1.
    """
    r = np.ones(num_clients, dtype=np.float64)
    r = r / r.sum()
    return r


def compute_reward_final_agreement(logits_k_nc: torch.Tensor) -> np.ndarray:
    """
    "Final-agreement only" scheme.
    - Compute final ensemble prediction via soft voting (mean prob)
    - For each sample, clients matching the final prediction share 1 equally
    - Final reward_i = sum of per-sample shares

    Labels are not used (can be applied at inference time).
    """
    K, N, C = logits_k_nc.shape
    probs_k_nc = F.softmax(logits_k_nc, dim=-1)  # [K, N, C]

    # Final soft ensemble prediction
    mean_prob_nc = probs_k_nc.mean(dim=0)  # [N, C]
    final_preds_n = mean_prob_nc.argmax(dim=-1)  # [N]

    # Per-client predictions
    client_preds_kn = probs_k_nc.argmax(dim=-1)  # [K, N]

    rewards = torch.zeros(K, dtype=torch.float64)
    for n in range(N):
        final_label = final_preds_n[n]
        winners = client_preds_kn[:, n] == final_label
        num_winners = int(winners.sum().item())
        if num_winners > 0:
            share = 1.0 / num_winners
            rewards[winners] += share

    return rewards.numpy()


def compute_reward_entropy_based(
    logits_k_nc: torch.Tensor, eps: float = 1e-12
) -> np.ndarray:
    """
    Entropy-based scheme.
    - For each sample t and client k, compute entropy H(p_{k,t})
    - w_{k,t} ∝ exp(-H) (lower entropy => higher weight)
    - Distribute reward 1 per sample proportional to w_{k,t}
    - Final reward_k = sum_t w_{k,t}

    Labels are not used.
    """
    K, N, C = logits_k_nc.shape
    probs_k_nc = F.softmax(logits_k_nc, dim=-1)  # [K, N, C]

    ent_k_n = -(probs_k_nc * (probs_k_nc + eps).log()).sum(dim=-1)  # [K, N]
    w_k_n = torch.exp(-ent_k_n)  # [K, N]
    w_k_n = w_k_n / (w_k_n.sum(dim=0, keepdim=True) + eps)  # per-sample normalize

    rewards = w_k_n.sum(dim=1).double()  # [K]
    return rewards.numpy()


# =============================
# Fairness: merit vs reward (L1)
# =============================


def merit_distribution(accs: np.ndarray) -> np.ndarray:
    """
    Ideal reward distribution proportional to accuracy: p_i = acc_i / sum_j acc_j
    """
    accs = np.asarray(accs, dtype=np.float64)
    accs = np.maximum(accs, 0.0)
    s = accs.sum()
    if s <= 0:
        return np.ones_like(accs) / len(accs)
    return accs / s


def reward_distribution(rewards: np.ndarray) -> np.ndarray:
    """
    Normalize raw rewards into a distribution r_i = reward_i / sum_j reward_j.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    rewards = np.maximum(rewards, 0.0)
    s = rewards.sum()
    if s <= 0:
        return np.ones_like(rewards) / len(rewards)
    return rewards / s


def skill_fairness(accs, rewards):
    """
    Skill-based fairness score in [0, 1] (higher is more score-proportional).

      - accs: per-client test accuracy (e.g., [39.6, 16.8, ...])
      - rewards: raw rewards for the scheme (e.g., counts)

    Steps:
      1) p = merit_distribution(accs)
      2) r = reward_distribution(rewards)
      3) D = L1(p, r) = sum_i |p_i - r_i|
      4) F = 1 - D/2  (0: very unfair, 1: perfectly fair)
    """
    p = merit_distribution(np.asarray(accs, dtype=np.float64))
    r = reward_distribution(np.asarray(rewards, dtype=np.float64))
    D = float(np.abs(p - r).sum())
    F = 1.0 - D / 2.0
    F = max(0.0, min(1.0, F))  # clamp to [0, 1]
    return F, D, p, r


def extract_labels_for_dirichlet(trainset, dataset_name: str) -> np.ndarray:
    """Extract a 1D label vector for Dirichlet split by dataset type."""
    dataset_name = dataset_name.lower()
    if dataset_name in ["cifar10", "cifar100", "fashionmnist", "emnist"]:
        # torchvision datasets use .targets
        labels = np.array(trainset.targets)
    elif dataset_name == "medmnist":
        # medmnist uses .labels (shape: [N, 1]) or .labels (N,)
        labels = np.array(trainset.labels)
        if labels.ndim > 1:
            labels = labels.squeeze()
        labels = labels.astype(int)
    else:
        raise ValueError(f"extract_labels_for_dirichlet not defined for {dataset_name}")
    return labels


# =============================
# Main
# =============================


def main():
    parser = argparse.ArgumentParser(
        description="Client Ensembling on CIFAR-10/100 with Dirichlet Non-IID + Reward Fairness"
    )
    parser.add_argument("--data-dir", type=str, default="./data", help="Data directory")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "cifar100", "emnist", "fashionmnist", "medmnist"],
        help="Dataset to use",
    )
    parser.add_argument(
        "--medmnist-subset",
        type=str,
        default=None,
        help="MedMNIST subset name (e.g., pathmnist, bloodmnist, ...)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        choices=["lenet", "resnet18"],
        help="Model architecture",
    )
    parser.add_argument("--num-clients", type=int, default=5)
    parser.add_argument(
        "--alpha", type=float, default=0.5, help="Dirichlet concentration"
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device, e.g., 'cuda:0', 'cpu', 'auto', 'mps'",
    )

    # Early stopping options
    parser.add_argument(
        "--early-stop",
        action="store_true",
        help="Use validation-based early stopping for local client training",
    )
    parser.add_argument(
        "--patience", type=int, default=5, help="Early stopping patience (epochs)"
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Fraction of each client's data used as validation set",
    )

    # model save/load options
    parser.add_argument(
        "--model-dir",
        type=str,
        default="./saved_models",
        help="Directory to save/load client models",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training and only evaluate using saved client models",
    )

    # wandb options
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Log results to Weights & Biases",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="federated-incentive",
        help="wandb project name",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default="thejungwon",
        help="wandb entity (username or team)",
    )

    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)

    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}", flush=True)
    print(f"Dataset: {args.dataset}, Model: {args.model}", flush=True)
    print(f"Eval-only: {args.eval_only}, Model dir: {args.model_dir}", flush=True)

    # wandb init
    run = None
    if args.use_wandb:
        if wandb is None:
            raise RuntimeError(
                "wandb is not installed, but --use-wandb is set. "
                "Install with `pip install wandb`."
            )
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            name=f"{args.dataset}_{args.model}_C{args.num_clients}_a{args.alpha}_seed{args.seed}",
            group=f"{args.dataset}_{args.model}",
        )

    # ---- Load data ----
    trainset, test_loader, num_classes, in_channels = get_dataloaders(
        args.data_dir,
        args.batch_size,
        args.dataset,
        num_workers=args.num_workers,
        medmnist_subset=args.medmnist_subset,
    )

    # ---- Dirichlet split ----
    # ---- Dirichlet split ----
    labels_np = extract_labels_for_dirichlet(trainset, args.dataset)
    client_indices = dirichlet_noniid_split(
        labels_np, num_clients=args.num_clients, alpha=args.alpha
    )
    client_loaders = make_client_loaders(
        trainset,
        client_indices,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # ---- Train or Load client models ----
    models = []
    for cid, loader in enumerate(client_loaders):
        ckpt_path = get_client_ckpt_path(args, cid)
        if args.eval_only:
            print(
                f"\n=== Loading client {cid} from checkpoint ===\n  {ckpt_path}",
                flush=True,
            )
            if not os.path.isfile(ckpt_path):
                raise FileNotFoundError(
                    f"Checkpoint for client {cid} not found at {ckpt_path}. "
                    "Run without --eval-only first to train & save models."
                )
            m = create_model(args.model, num_classes=num_classes)
            state = torch.load(ckpt_path, map_location="cpu")
            m.load_state_dict(state)
            models.append(m)
        else:
            print(
                f"\n=== Training client {cid} | samples: {len(loader.dataset)} "
                f"({args.dataset}, {args.model}) ===",
                flush=True,
            )
            m = create_model(args.model, num_classes=num_classes)
            m = train_one_client(
                m,
                loader,
                device=device,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                early_stop=args.early_stop,
                patience=args.patience,
                val_fraction=args.val_fraction,
            )
            print(f"  [Save] Saving client {cid} model to {ckpt_path}", flush=True)
            torch.save(m.state_dict(), ckpt_path)
            models.append(m)

    # ---- Evaluation: collect logits ----
    print("\nCollecting logits on test set (no TTA)...", flush=True)
    logits, labels = collect_logits(models, test_loader, device)

    K, N, C = logits.shape
    preds_kn = logits.argmax(dim=-1)  # [K, N]

    # Oracle: per-sample best client (upper bound)
    best_any = (preds_kn == labels.unsqueeze(0)).any(dim=0)  # [N]
    oracle_acc = 100.0 * best_any.float().mean().item()
    print(
        "Oracle per-sample best-client accuracy:",
        round(oracle_acc, 2),
        flush=True,
    )

    # Per-client accuracies
    print("\nPer-client test accuracy (no TTA):", flush=True)
    client_accs = []
    for i in range(len(models)):
        acc_i = accuracy_from_preds(preds_kn[i], labels)
        client_accs.append(acc_i)
        print(f"  Client {i}: {acc_i:.2f}%", flush=True)

    client_accs_np = np.array(client_accs, dtype=np.float64)

    # ---- Ensemble results (no TTA) ----
    hv_acc = acc_from_logits(logits, labels, method="hard")
    sv_acc = acc_from_logits(logits, labels, method="soft")
    ew_acc = acc_from_logits(logits, labels, method="entropy")

    print(
        f"\n=== Ensemble Results on {args.dataset.upper()} Test (no TTA) ===",
        flush=True,
    )
    print(f"Hard voting:                  {hv_acc:.2f}%", flush=True)
    print(f"Soft voting (prob avg):       {sv_acc:.2f}%", flush=True)
    print(f"Entropy-weighted soft voting: {ew_acc:.2f}%", flush=True)

    # =========================
    # Reward allocation & fairness
    # =========================
    print("\n=== Reward allocation & merit fairness analysis ===", flush=True)
    rewards_equal = compute_reward_equal(K)
    rewards_majority = compute_reward_final_agreement(logits)
    rewards_entropy = compute_reward_entropy_based(logits)

    reward_schemes = {
        "equal_1_over_n": rewards_equal,
        "final_agreement": rewards_majority,
        "entropy_based": rewards_entropy,
    }

    # Merit distribution (ideal score-proportional allocation)
    merit_p = merit_distribution(client_accs_np)
    print("Merit distribution (from accuracies):", flush=True)
    for i, (acc_i, p_i) in enumerate(zip(client_accs_np, merit_p)):
        print(
            f"  client {i}: acc = {acc_i:.2f}%, merit p_i = {p_i:.4f}",
            flush=True,
        )

    # Storage for results
    fairness_results = {}

    for name, r in reward_schemes.items():
        rewards_np = np.asarray(r, dtype=np.float64)

        F_skill, D_l1, p_dist, r_dist = skill_fairness(client_accs_np, rewards_np)
        fairness_results[name] = {
            "F_skill": F_skill,
            "D_l1": D_l1,
        }

        print(f"\n[{name}] raw rewards per client:", flush=True)
        for i, ri in enumerate(r):
            print(f"  client {i}: reward = {ri:.4f}", flush=True)

        print("  Merit (ideal) distribution p_i (from acc):", flush=True)
        for i, pi in enumerate(p_dist):
            print(f"    client {i}: p_i = {pi:.4f}", flush=True)

        print("  Actual reward distribution r_i (normalized):", flush=True)
        for i, ri in enumerate(r_dist):
            print(f"    client {i}: r_i = {ri:.4f}", flush=True)

        print(
            f"  -> Skill-based fairness F_skill = {F_skill:.4f} (0~1, higher = more 'score-proportional')",
            flush=True,
        )
        print(
            f"     (L1 distance D(p,r) = {D_l1:.4f}, 0=perfect, max=2)",
            flush=True,
        )

    # ---- wandb logging ----
    if run is not None:
        log_dict = {}

        # per-client acc
        for i, acc_i in enumerate(client_accs):
            log_dict[f"client_test_acc/client_{i}"] = acc_i
        log_dict["client_test_acc/max"] = float(np.max(client_accs))
        log_dict["client_test_acc/min"] = float(np.min(client_accs))
        log_dict["client_test_acc/mean"] = float(np.mean(client_accs))
        log_dict["client_test_acc/std"] = float(np.std(client_accs))

        # oracle & ensembles
        log_dict["oracle/per_sample_best_client_acc"] = oracle_acc
        log_dict["ensemble/hard_voting_acc"] = hv_acc
        log_dict["ensemble/soft_voting_acc"] = sv_acc
        log_dict["ensemble/entropy_weighted_soft_voting_acc"] = ew_acc

        # reward + fairness
        for name, r in reward_schemes.items():
            rewards_np = np.asarray(r, dtype=np.float64)
            F_skill, D_l1, _, _ = skill_fairness(client_accs_np, rewards_np)

            for i, ri in enumerate(r):
                log_dict[f"reward/{name}/client_{i}"] = float(ri)
            log_dict[f"fairness/{name}/F_skill"] = float(F_skill)
            log_dict[f"fairness/{name}/D_l1"] = float(D_l1)

        # meta
        log_dict["meta/num_clients"] = args.num_clients
        log_dict["meta/alpha"] = args.alpha
        log_dict["meta/seed"] = args.seed
        log_dict["meta/dataset"] = args.dataset
        log_dict["meta/model"] = args.model

        wandb.log(log_dict)
        run.finish()


if __name__ == "__main__":
    main()
