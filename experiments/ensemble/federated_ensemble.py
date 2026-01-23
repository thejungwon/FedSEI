#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Federated-style independent client training on CIFAR-10/CIFAR-100/EMNIST/FashionMNIST/MedMNIST
using a Dirichlet Non-IID split, then ensemble inference with:
  1) Hard voting
  2) Soft voting (probability averaging)
  3) Entropy-based weighted soft voting (per-sample weights)
  4) (Optional) TTA-based soft ensemble
  5) (Optional) TTA-based MI-weighted ensemble

Usage (example):

  # CIFAR-10
  python federated_ensemble.py --dataset cifar10 --model resnet18 \
      --num-clients 5 --alpha 0.5 --epochs 5 --batch-size 128 --lr 0.001

  # FashionMNIST
  python federated_ensemble.py --dataset fashionmnist --model lenet \
      --num-clients 5 --alpha 0.5 --epochs 5 --batch-size 128 --lr 0.001

  # EMNIST (balanced split)
  python federated_ensemble.py --dataset emnist --model resnet18 \
      --num-clients 5 --alpha 0.5 --epochs 5 --batch-size 128 --lr 0.001

  # MedMNIST (e.g., PathMNIST)
  python federated_ensemble.py --dataset medmnist --medmnist-subset pathmnist \
      --model resnet18 --num-clients 5 --alpha 0.5 --epochs 5 --batch-size 128 --lr 0.001
"""
import matplotlib.pyplot as plt

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
from torchvision.transforms import functional as TF

# wandb (optional)
try:
    import wandb
except ImportError:
    wandb = None

# medmnist (optional)
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

# ----- LeNet -----


class LeNet(nn.Module):
    """Classic LeNet-5 style network adapted for 32x32x3 input."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)  # 32 -> 28
        self.pool = nn.MaxPool2d(2, 2)  # 28 -> 14, 10 -> 5
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


# ----- ResNet18 (CIFAR/MNIST/MedMNIST version) -----


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
    def __init__(self, block, num_blocks, num_classes=10, in_channels=3):
        super().__init__()
        self.in_planes = 64

        # 3x3 conv, no maxpool, flexible in_channels
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
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


def ResNet18(num_classes=10, in_channels=3):
    return ResNet(
        BasicBlock, [2, 2, 2, 2], num_classes=num_classes, in_channels=in_channels
    )


# ----- Model factory -----


def create_model(model_name: str, num_classes: int, in_channels: int = 3) -> nn.Module:
    model_name = model_name.lower()
    if model_name == "lenet":
        # LeNet expects 3-channel 32x32 input; MNIST-like datasets are expanded to 3 channels in transforms.
        return LeNet(num_classes=num_classes)
    elif model_name == "resnet18":
        return ResNet18(num_classes=num_classes, in_channels=in_channels)
    else:
        raise ValueError(f"Unknown model: {model_name}")


# =============================
# Dirichlet Non-IID splitter
# =============================


def dirichlet_noniid_split(
    labels: np.ndarray, num_clients: int, alpha: float
) -> List[np.ndarray]:
    """Split indices into Non-IID partitions using a class-wise Dirichlet.

    For each class c, draw a Dirichlet(num_clients, alpha) over clients to allocate samples of class c.
    """
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
    log_interval: int = 50,
):
    """
    Train a single client model with optional validation-based early stopping.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # ---- Optional client-wise train/val split ----
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
        # ---- Train ----
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
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

            if log_interval > 0 and (batch_idx + 1) % log_interval == 0:
                avg_loss_so_far = running_loss / max(1, total)
                acc_so_far = 100.0 * correct / max(1, total)
                print(
                    f"    [Batch {batch_idx + 1:04d}/{len(train_loader):04d}] "
                    f"loss={avg_loss_so_far:.4f} acc={acc_so_far:.2f}%",
                    flush=True,
                )

        avg_loss = running_loss / max(1, total)
        train_acc = 100.0 * correct / max(1, total)

        # ---- Validation ----
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

            # early stopping check (val loss)
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
    """Return (logits, labels)
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

    logits = torch.cat(all_logits, dim=1)  # [M, N, C]
    labels = torch.cat(all_labels, dim=0)  # [N]
    return logits, labels


def accuracy_from_preds(preds: torch.Tensor, labels: torch.Tensor) -> float:
    correct = (preds == labels).sum().item()
    return 100.0 * correct / labels.numel()


def compute_confusion_matrix(
    y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int
) -> np.ndarray:
    """
    Simple confusion matrix: rows = true class, cols = predicted class.
    """
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true_np, y_pred_np):
        cm[t, p] += 1
    return cm


# =============================
# Ensemble Methods (no TTA)
# =============================


def acc_from_logits(
    logits_k_nc: torch.Tensor,
    labels: torch.Tensor,
    method: str = "soft",
    eps: float = 1e-12,
) -> float:
    """Generic ensembling over K predictors with logits shape [K, N, C].
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
        # per-model, per-sample entropy
        ent = -(probs * (probs + eps).log()).sum(dim=-1)  # [K, N]
        w = torch.exp(-ent)  # low entropy -> high weight
        w = w / (w.sum(dim=0, keepdim=True) + eps)
        weighted = (w.unsqueeze(-1) * probs).sum(dim=0)  # [N, C]
        final = weighted.argmax(dim=-1)
        return accuracy_from_preds(final, labels)
    else:
        raise ValueError(f"Unknown method: {method}")


# =============================
# Unsupervised Global & Local Ensemble (no TTA)
# =============================


def compute_spectral_global_weights_from_logits(
    logits_k_nc: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    SUMMA-style 간단 global reliability weight 추정.
    """
    K, N, C = logits_k_nc.shape

    # [K, N, C] -> [K, N]: max prob (confidence) per sample
    probs_k_nc = F.softmax(logits_k_nc, dim=-1)
    scores_k_n = probs_k_nc.max(dim=-1).values  # [K, N]

    # center per client
    scores_centered = scores_k_n - scores_k_n.mean(dim=1, keepdim=True)  # [K, N]

    if N <= 1:
        alpha = torch.full((K,), 1.0 / K)
        return alpha

    Cmat = scores_centered @ scores_centered.t()
    Cmat = Cmat / max(1, (N - 1))

    evals, evecs = torch.linalg.eigh(Cmat)
    idx_max = torch.argmax(evals)
    v = evecs[:, idx_max]

    alpha = v.abs()
    alpha = alpha / (alpha.sum() + eps)
    return alpha  # [K]


def spectral_global_ensemble_from_logits(
    logits_k_nc: torch.Tensor,
    labels: torch.Tensor,
    eps: float = 1e-12,
) -> Tuple[float, torch.Tensor, torch.Tensor]:
    K, N, C = logits_k_nc.shape
    probs_k_nc = F.softmax(logits_k_nc, dim=-1)  # [K, N, C]

    alpha_k = compute_spectral_global_weights_from_logits(logits_k_nc, eps=eps)  # [K]
    weighted_probs_nc = (alpha_k.view(K, 1, 1) * probs_k_nc).sum(dim=0)  # [N, C]
    preds = weighted_probs_nc.argmax(dim=-1)
    acc = accuracy_from_preds(preds, labels)
    return acc, alpha_k, preds


def consensus_js_ensemble_from_logits(
    logits_k_nc: torch.Tensor,
    labels: torch.Tensor,
    beta: float = 5.0,
    eps: float = 1e-12,
) -> float:
    """
    Cross-client JS-based local consensus weighting.
    """
    K, N, C = logits_k_nc.shape
    probs_k_nc = F.softmax(logits_k_nc, dim=-1).clamp_min(eps)  # [K, N, C]

    # consensus distribution m_n: [N, C]
    m_nc = probs_k_nc.mean(dim=0).clamp_min(eps)  # [N, C]
    log_m_nc = m_nc.log()  # [N, C]

    log_p_k_nc = probs_k_nc.log()  # [K, N, C]

    # KL(p_k || m)
    kl_p_m = (probs_k_nc * (log_p_k_nc - log_m_nc.unsqueeze(0))).sum(dim=-1)  # [K, N]
    # KL(m || p_k)
    kl_m_p = (m_nc.unsqueeze(0) * (log_m_nc.unsqueeze(0) - log_p_k_nc)).sum(
        dim=-1
    )  # [K, N]

    js_k_n = 0.5 * (kl_p_m + kl_m_p)  # [K, N]

    # Larger JS means farther from consensus -> downweight: w ∝ exp(-beta * JS)
    w_k_n = torch.exp(-beta * js_k_n)  # [K, N]
    w_k_n = w_k_n / (w_k_n.sum(dim=0, keepdim=True) + eps)  # normalize over K

    weighted_probs_nc = (w_k_n.unsqueeze(-1) * probs_k_nc).sum(dim=0)  # [N, C]
    preds = weighted_probs_nc.argmax(dim=-1)
    acc = accuracy_from_preds(preds, labels)
    return acc


def hybrid_spectral_consensus_ensemble_from_logits(
    logits_k_nc: torch.Tensor,
    labels: torch.Tensor,
    alpha_global: torch.Tensor,
    beta: float = 5.0,
    eps: float = 1e-12,
) -> float:
    """
    Global (spectral) weight × Local (JS-consensus) hybrid.
    """
    K, N, C = logits_k_nc.shape
    probs_k_nc = F.softmax(logits_k_nc, dim=-1).clamp_min(eps)  # [K, N, C]

    # consensus distribution m_n: [N, C]
    m_nc = probs_k_nc.mean(dim=0).clamp_min(eps)  # [N, C]
    log_m_nc = m_nc.log()  # [N, C]

    log_p_k_nc = probs_k_nc.log()  # [K, N, C]

    # KL(p_k || m)
    kl_p_m = (probs_k_nc * (log_p_k_nc - log_m_nc.unsqueeze(0))).sum(dim=-1)  # [K, N]
    # KL(m || p_k)
    kl_m_p = (m_nc.unsqueeze(0) * (log_m_nc.unsqueeze(0) - log_p_k_nc)).sum(
        dim=-1
    )  # [K, N]
    js_k_n = 0.5 * (kl_p_m + kl_m_p)  # [K, N]

    # local part: g_{k,n}
    g_k_n = torch.exp(-beta * js_k_n)  # [K, N]

    # global part: alpha_k
    alpha_k = alpha_global.view(K, 1)  # [K, 1]

    # hybrid: w_{k,n} ∝ alpha_k * g_{k,n}
    w_k_n = alpha_k * g_k_n  # [K, N]
    w_k_n = w_k_n / (w_k_n.sum(dim=0, keepdim=True) + eps)

    weighted_probs_nc = (w_k_n.unsqueeze(-1) * probs_k_nc).sum(dim=0)  # [N, C]
    preds = weighted_probs_nc.argmax(dim=-1)
    acc = accuracy_from_preds(preds, labels)
    return acc


# =============================
# TTA helpers & TTA-based ensemble
# =============================


def tta_random_augment_batch(x: torch.Tensor, angle: float = 10.0) -> torch.Tensor:
    """
    Simple label-preserving augmentation on a batch of normalized tensors.
    - Small random rotation in [-angle, angle] degrees
    x: [B, C, H, W] (on CPU)
    """
    B = x.size(0)
    out = []
    for i in range(B):
        img = x[i]
        theta = random.uniform(-angle, angle)
        img = TF.rotate(img, theta)
        out.append(img)
    return torch.stack(out, dim=0)


def tta_mi_weighted_ensemble(
    models: List[nn.Module],
    loader: DataLoader,
    device,
    tta_steps: int = 4,
    tta_angle: float = 10.0,
    gamma: float = 1.0,
    eps: float = 1e-12,
    top_k: int = None,
):
    """
    TTA 기반 ensemble with multiple instability metrics.
    """
    for m in models:
        m.eval()
        m.to(device)

    K = len(models)
    all_mean_probs_batches = []  # list of [K, B, C]
    all_l2_batches = []  # list of [K, B]
    all_cos_batches = []  # list of [K, B]
    all_kl_batches = []  # list of [K, B]
    all_js_batches = []  # list of [K, B]
    all_labels = []

    with torch.no_grad():
        for inputs, targets in loader:
            # inputs: [B, C, H, W] on CPU
            B = inputs.size(0)
            targets = _normalize_targets_for_ce(targets)
            all_labels.append(targets.clone())

            # ---- 1) Original logits and prob (p0) ----
            inputs_dev = inputs.to(device)
            logits0_list = []
            for m in models:
                logits0 = m(inputs_dev)  # [B, C]
                logits0_list.append(logits0.detach().cpu())
            logits0_k_bc = torch.stack(logits0_list, dim=0)  # [K, B, C]

            K_, B_, C_ = logits0_k_bc.shape
            assert K_ == K and B_ == B

            # p0: [K, B, C]
            probs0_k_bc = F.softmax(logits0_k_bc, dim=-1).clamp_min(eps)

            # Sample-wise top-k index (per client)
            topk_idx_k_bc = None
            if (top_k is not None) and (top_k > 0) and (top_k < C_):
                topk_idx_k_bc = torch.empty(K, B, top_k, dtype=torch.long)
                for k in range(K):
                    _, idx = logits0_k_bc[k].topk(top_k, dim=-1)  # [B, top_k]
                    topk_idx_k_bc[k] = idx

            # ---- 2) TTA loop: mean prob + metric instability ----
            sum_probs_k_bc = torch.zeros_like(logits0_k_bc)  # [K, B, C]
            sum_l2_k_b = torch.zeros(K, B, dtype=logits0_k_bc.dtype)  # [K, B]
            sum_cos_k_b = torch.zeros(K, B, dtype=logits0_k_bc.dtype)  # [K, B]
            sum_kl_k_b = torch.zeros(K, B, dtype=logits0_k_bc.dtype)  # [K, B]
            sum_js_k_b = torch.zeros(K, B, dtype=logits0_k_bc.dtype)  # [K, B]

            for _ in range(tta_steps):
                aug = tta_random_augment_batch(
                    inputs, angle=tta_angle
                )  # [B, C, H, W], CPU
                aug_dev = aug.to(device)

                batch_probs_list = []
                for k, m in enumerate(models):
                    logits_aug = m(aug_dev)  # [B, C] on device
                    logits_aug_cpu = logits_aug.detach().cpu()  # [B, C]

                    probs_t_full = F.softmax(logits_aug_cpu, dim=-1).clamp_min(
                        eps
                    )  # [B, C]
                    batch_probs_list.append(probs_t_full)

                    # ===== Apply top-k to instability metrics =====
                    if topk_idx_k_bc is not None:
                        idx_top = topk_idx_k_bc[k]  # [B, top_k]

                        z0_sub = logits0_k_bc[k].gather(1, idx_top)  # [B, top_k]
                        zt_sub = logits_aug_cpu.gather(1, idx_top)  # [B, top_k]

                        p0_sub = (
                            probs0_k_bc[k].gather(1, idx_top).clamp_min(eps)
                        )  # [B, top_k]
                        pt_sub = probs_t_full.gather(1, idx_top).clamp_min(
                            eps
                        )  # [B, top_k]
                        p0_sub = p0_sub / p0_sub.sum(dim=-1, keepdim=True)
                        pt_sub = pt_sub / pt_sub.sum(dim=-1, keepdim=True)
                    else:
                        z0_sub = logits0_k_bc[k]  # [B, C]
                        zt_sub = logits_aug_cpu  # [B, C]
                        p0_sub = probs0_k_bc[k]  # [B, C]
                        pt_sub = probs_t_full  # [B, C]

                    # ---------- L2 instability ----------
                    diff = zt_sub - z0_sub  # [B, d]
                    l2 = diff.pow(2).sum(dim=-1).sqrt()  # [B]
                    sum_l2_k_b[k] += l2

                    # ---------- Cosine distance ----------
                    cos = F.cosine_similarity(z0_sub, zt_sub, dim=-1, eps=1e-8)  # [B]
                    cos_dist = 1.0 - cos
                    sum_cos_k_b[k] += cos_dist

                    # ---------- KL(p0_sub || pt_sub) ----------
                    kl = (p0_sub * (p0_sub.log() - pt_sub.log())).sum(dim=-1)  # [B]
                    sum_kl_k_b[k] += kl

                    # ---------- JS(p0_sub || pt_sub) ----------
                    m_prob = 0.5 * (p0_sub + pt_sub)  # [B, d]
                    m_prob = m_prob.clamp_min(eps)
                    kl_p0_m = (p0_sub * (p0_sub.log() - m_prob.log())).sum(
                        dim=-1
                    )  # [B]
                    kl_pt_m = (pt_sub * (pt_sub.log() - m_prob.log())).sum(
                        dim=-1
                    )  # [B]
                    js = 0.5 * (kl_p0_m + kl_pt_m)  # [B]
                    sum_js_k_b[k] += js

                batch_probs = torch.stack(batch_probs_list, dim=0)  # [K, B, C]
                sum_probs_k_bc += batch_probs

            mean_probs_k_bc = sum_probs_k_bc / float(tta_steps)  # [K, B, C]
            l2_k_b = sum_l2_k_b / float(tta_steps)  # [K, B]
            cos_k_b = sum_cos_k_b / float(tta_steps)  # [K, B]
            kl_k_b = sum_kl_k_b / float(tta_steps)  # [K, B]
            js_k_b = sum_js_k_b / float(tta_steps)  # [K, B]

            all_mean_probs_batches.append(mean_probs_k_bc)
            all_l2_batches.append(l2_k_b)
            all_cos_batches.append(cos_k_b)
            all_kl_batches.append(kl_k_b)
            all_js_batches.append(js_k_b)

    # ---- Concatenate batches ----
    mean_probs_k_nc = torch.cat(all_mean_probs_batches, dim=1)  # [K, N, C]
    l2_k_n = torch.cat(all_l2_batches, dim=1)  # [K, N]
    cos_k_n = torch.cat(all_cos_batches, dim=1)  # [K, N]
    kl_k_n = torch.cat(all_kl_batches, dim=1)  # [K, N]
    js_k_n = torch.cat(all_js_batches, dim=1)  # [K, N]
    labels = torch.cat(all_labels, dim=0)  # [N]

    # ---- TTA-soft ensemble (baseline) ----
    tta_soft_mean_prob_nc = mean_probs_k_nc.mean(dim=0)  # [N, C]
    tta_soft_preds = tta_soft_mean_prob_nc.argmax(dim=-1)
    tta_soft_acc = accuracy_from_preds(tta_soft_preds, labels)

    # ---- Metric-weighted ensemble ----
    def metric_weighted_acc(instab_k_n: torch.Tensor, return_preds: bool = False):
        w_k_n = torch.exp(gamma * instab_k_n)
        w_k_n = w_k_n / (w_k_n.sum(dim=0, keepdim=True) + eps)  # normalize over K
        weighted_probs_nc = (w_k_n.unsqueeze(-1) * mean_probs_k_nc).sum(dim=0)  # [N, C]
        preds = weighted_probs_nc.argmax(dim=-1)
        acc = accuracy_from_preds(preds, labels)
        if return_preds:
            return acc, preds
        return acc

    acc_l2, preds_l2 = metric_weighted_acc(l2_k_n, return_preds=True)
    acc_cos = metric_weighted_acc(cos_k_n)
    acc_kl = metric_weighted_acc(kl_k_n)
    acc_js = metric_weighted_acc(js_k_n)

    return tta_soft_acc, acc_l2, acc_cos, acc_kl, acc_js, preds_l2


# =============================
# Main
# =============================


def main():
    parser = argparse.ArgumentParser(
        description="Client Ensembling on CIFAR-10/100/EMNIST/FashionMNIST/MedMNIST with Dirichlet Non-IID"
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
        "--log-interval",
        type=int,
        default=50,
        help="Print training stats every N batches (0 disables).",
    )
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

    # TTA options
    parser.add_argument(
        "--tta-steps",
        type=int,
        default=0,
        help="Number of TTA samples per test image (0 disables TTA-based ensemble).",
    )
    parser.add_argument(
        "--tta-mi-gamma",
        type=float,
        default=1.0,
        help="Gamma for MI-based TTA weighting (w ∝ exp(gamma * instability)).",
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
        default="federated-ensemble",
        help="wandb project name",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="wandb entity (username or team)",
    )
    parser.add_argument(
        "--tta-angle",
        type=float,
        default=5.0,
        help="Angle for TTA random rotation",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-k for instability metrics",
    )
    parser.add_argument(
        "--consensus-beta",
        type=float,
        default=5.0,
        help="Temperature for JS-based consensus weighting (larger = more peaky).",
    )

    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)

    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}", flush=True)
    print(
        f"Dataset: {args.dataset}"
        + (f" (subset={args.medmnist_subset})" if args.dataset == "medmnist" else "")
        + f", Model: {args.model}",
        flush=True,
    )
    print(f"Eval-only: {args.eval_only}, Model dir: {args.model_dir}", flush=True)
    print(
        f"TTA steps: {args.tta_steps}, TTA MI gamma: {args.tta_mi_gamma}",
        flush=True,
    )

    # ---- wandb init ----
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
            name=(
                f"{args.dataset}"
                + (f"-{args.medmnist_subset}" if args.dataset == "medmnist" else "")
                + f"_{args.model}_C{args.num_clients}_a{args.alpha}_seed{args.seed}"
            ),
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
    print("Splitting data among clients with Dirichlet non-IID...", flush=True)
    labels_np = extract_labels_for_dirichlet(trainset, args.dataset)
    client_indices = dirichlet_noniid_split(
        labels_np, num_clients=args.num_clients, alpha=args.alpha
    )
    sizes = [len(idxs) for idxs in client_indices]
    print(
        "Client sample counts: "
        + ", ".join(f"{i}:{n}" for i, n in enumerate(sizes)),
        flush=True,
    )
    client_loaders = make_client_loaders(
        trainset,
        client_indices,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print("Client DataLoaders ready.", flush=True)

    # ---- Train or Load client models ----
    models = []
    for cid, loader in enumerate(client_loaders):
        ckpt_path = get_client_ckpt_path(args, cid)
        if args.eval_only:
            # load existing model
            print(
                f"\n=== Loading client {cid} from checkpoint ===\n  {ckpt_path}",
                flush=True,
            )
            if not os.path.isfile(ckpt_path):
                raise FileNotFoundError(
                    f"Checkpoint for client {cid} not found at {ckpt_path}. "
                    "Run without --eval-only first to train & save models."
                )
            m = create_model(
                args.model, num_classes=num_classes, in_channels=in_channels
            )
            state = torch.load(ckpt_path, map_location="cpu")
            m.load_state_dict(state)
            models.append(m)
        else:
            # train new model and save
            print(
                f"\n=== Training client {cid} | samples: {len(loader.dataset)} "
                f"({args.dataset}, {args.model}) ===",
                flush=True,
            )
            m = create_model(
                args.model, num_classes=num_classes, in_channels=in_channels
            )
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
                log_interval=args.log_interval,
            )
            print(f"  [Save] Saving client {cid} model to {ckpt_path}", flush=True)
            torch.save(m.state_dict(), ckpt_path)
            models.append(m)

    # ---- Evaluation (no TTA) ----
    print("\nCollecting logits on test set (no TTA)...", flush=True)
    logits, labels = collect_logits(models, test_loader, device)

    K, N, C = logits.shape
    preds = logits.argmax(dim=-1)  # [K, N]

    # Oracle: per-sample best client (upper bound)
    best_any = (preds == labels.unsqueeze(0)).any(dim=0)  # [N]
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
        acc_i = accuracy_from_preds(logits[i].argmax(dim=-1), labels)
        client_accs.append(acc_i)
        print(f"  Client {i}: {acc_i:.2f}%", flush=True)

    # Ensemble results (no TTA)
    hv_acc = acc_from_logits(logits, labels, method="hard")
    sv_acc = acc_from_logits(logits, labels, method="soft")
    ew_acc = acc_from_logits(logits, labels, method="entropy")

    print(
        f"\n=== Ensemble Results on {args.dataset.upper()} Test (no TTA) ===",
        flush=True,
    )
    print("-- Across clients (no TTA) --", flush=True)
    print(f"Hard voting:                     {hv_acc:.2f}%", flush=True)
    print(f"Soft voting (prob avg):          {sv_acc:.2f}%", flush=True)
    print(f"Entropy-weighted soft voting:    {ew_acc:.2f}%", flush=True)

    # ---- Unsupervised global / local ensemble (no TTA) ----
    spectral_acc, alpha_global, spectral_preds = spectral_global_ensemble_from_logits(
        logits, labels
    )

    consensus_acc = consensus_js_ensemble_from_logits(
        logits, labels, beta=args.consensus_beta
    )
    hybrid_acc = hybrid_spectral_consensus_ensemble_from_logits(
        logits, labels, alpha_global=alpha_global, beta=args.consensus_beta
    )

    print(
        f"Spectral global-weighted soft voting:      {spectral_acc:.2f}%",
        flush=True,
    )
    print(
        f"JS-consensus local-weighted ensemble:      {consensus_acc:.2f}%",
        flush=True,
    )
    print(
        f"Hybrid (global × local) weighted ensemble: {hybrid_acc:.2f}%",
        flush=True,
    )

    # ---- TTA-based ensemble (optional) ----
    tta_soft_acc = None
    acc_l2 = acc_cos = acc_kl = acc_js = None
    l2_tta_preds = None
    if args.tta_steps > 0:
        print(
            f"\nRunning TTA-based ensemble with T={args.tta_steps}, gamma={args.tta_mi_gamma}...",
            flush=True,
        )
        (
            tta_soft_acc,
            acc_l2,
            acc_cos,
            acc_kl,
            acc_js,
            l2_tta_preds,
        ) = tta_mi_weighted_ensemble(
            models,
            test_loader,
            device=device,
            tta_steps=args.tta_steps,
            tta_angle=args.tta_angle,
            gamma=args.tta_mi_gamma,
            top_k=args.top_k,
        )
        print(
            f"\n=== TTA-based Ensemble Results on {args.dataset.upper()} Test ===",
            flush=True,
        )
        print(
            f"TTA-soft ensemble (avg over K,T):         {tta_soft_acc:.2f}%",
            flush=True,
        )
        print(f"TTA L2-weighted ensemble:                 {acc_l2:.2f}%", flush=True)
        print(
            f"TTA Cosine-weighted ensemble:             {acc_cos:.2f}%",
            flush=True,
        )
        print(f"TTA KL-div-weighted ensemble:             {acc_kl:.2f}%", flush=True)
        print(f"TTA JS-div-weighted ensemble:             {acc_js:.2f}%", flush=True)

    # ---- wandb logging ----
    if run is not None:
        log_dict = {}

        # Per-client accuracy
        for i, acc_i in enumerate(client_accs):
            log_dict[f"client_test_acc/client_{i}"] = acc_i

        # Client statistics
        log_dict["client_test_acc/max"] = float(np.max(client_accs))
        log_dict["client_test_acc/min"] = float(np.min(client_accs))
        log_dict["client_test_acc/mean"] = float(np.mean(client_accs))
        log_dict["client_test_acc/std"] = float(np.std(client_accs))

        # Oracle
        log_dict["oracle/per_sample_best_client_acc"] = oracle_acc

        # Ensemble results (no TTA)
        log_dict["ensemble/hard_voting_acc"] = hv_acc
        log_dict["ensemble/soft_voting_acc"] = sv_acc
        log_dict["ensemble/entropy_weighted_soft_voting_acc"] = ew_acc
        # Unsupervised global/local ensemble (no TTA)
        log_dict["ensemble/spectral_global_acc"] = spectral_acc
        log_dict["ensemble/consensus_js_acc"] = consensus_acc
        log_dict["ensemble/hybrid_spectral_consensus_acc"] = hybrid_acc

        # global weight
        for k, ak in enumerate(alpha_global.tolist()):
            log_dict[f"ensemble/global_weight_client_{k}"] = ak

        # TTA results (optional)
        if tta_soft_acc is not None:
            log_dict["ensemble_tta/soft_acc"] = tta_soft_acc
        if acc_l2 is not None:
            log_dict["ensemble_tta/l2_weighted_acc"] = acc_l2
        if acc_cos is not None:
            log_dict["ensemble_tta/cosine_weighted_acc"] = acc_cos
        if acc_kl is not None:
            log_dict["ensemble_tta/kl_weighted_acc"] = acc_kl
        if acc_js is not None:
            log_dict["ensemble_tta/js_weighted_acc"] = acc_js

        # Metadata (for table filtering)
        log_dict["meta/num_clients"] = args.num_clients
        log_dict["meta/alpha"] = args.alpha
        log_dict["meta/seed"] = args.seed
        log_dict["meta/dataset"] = args.dataset
        if args.dataset == "medmnist":
            log_dict["meta/medmnist_subset"] = args.medmnist_subset
        log_dict["meta/model"] = args.model

        wandb.log(log_dict)

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
