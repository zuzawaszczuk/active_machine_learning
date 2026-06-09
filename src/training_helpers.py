from collections import defaultdict
import numpy as np
from torch.utils.data import random_split
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader, Subset, Dataset, TensorDataset
from sklearn.model_selection import train_test_split
from torchvision.models import ResNet18_Weights
from torchvision import datasets, transforms


def imbalance_train_dataset(train_cifar, seed=42):
    rng = np.random.default_rng(seed)
    targets = np.array(train_cifar.targets)

    print("Original length of training dataset:", len(train_cifar))
    
    unique_classes = np.unique(targets)
    minority_classes = set(range(20))
    selected_indices = []

    for cls in unique_classes:
        idxs = np.flatnonzero(targets == cls)
        
        if len(idxs) == 0:
            continue
            
        if cls in minority_classes:
            sampled = idxs
        else:
            sampled = rng.choice(idxs, size=max(1, len(idxs) // 10), replace=False)
            
        selected_indices.append(sampled)

    selected_indices = np.concatenate(selected_indices)
    rng.shuffle(selected_indices)

    print("Length of training dataset after imbalance sampling:", len(selected_indices))
    
    return Subset(train_cifar, selected_indices)


def get_dataset() -> tuple[Dataset, Dataset, Dataset]:
    train_cifar = datasets.CIFAR100(root="./data", train=True, download=True, transform=transforms.ToTensor())
    test_cifar = datasets.CIFAR100(root="./data", train=False, download=True, transform=transforms.ToTensor())

    val_size = 5000
    test_size = len(test_cifar) - val_size

    val_cifar, test_cifar = random_split(test_cifar, [val_size, test_size])
    return imbalance_train_dataset(train_cifar), val_cifar, test_cifar


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 100):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


def get_model() -> nn.Module:
    # model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    # model.conv1 = nn.Conv2d(
    #     in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False
    # )
    # model.maxpool = nn.Identity()
    # model.fc = nn.Linear(model.fc.in_features, 10)
    # return SimpleCNN(num_classes=100)
    return SimpleCNN(num_classes=100)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
):
    model.train()

    for (x, y, idx) in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for (x, y, idx) in loader:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            preds = logits.argmax(dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total
