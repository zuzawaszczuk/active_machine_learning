from torchvision import datasets, transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.model_selection import train_test_split
from torchvision.models import ResNet18_Weights


def get_dataset() -> tuple[Dataset, Dataset, Dataset]:
    train_cifar = datasets.CIFAR10(root="./data", train=True, download=True, transform=transforms.ToTensor())
    test_cifar = datasets.CIFAR10(root="./data", train=False, download=True, transform=transforms.ToTensor())

    train_idx, val_idx = train_test_split(
        range(len(train_cifar)),
        test_size=5000,
        random_state=42,
    )
    train_dataset = Subset(train_cifar, train_idx)
    val_dataset = Subset(train_cifar, val_idx)

    return train_dataset, val_dataset, test_cifar


def get_model() -> nn.Module:
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.conv1 = nn.Conv2d(
        in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False
    )
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model


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
