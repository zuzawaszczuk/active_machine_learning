from torchvision import datasets, transforms
from dataset import PoolDataset
from controller import ActiveLearningController
from methods import UncertaintySampling
from methods import LeastConfidenceSampling
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights


def get_dataset() -> PoolDataset:
    cifar = datasets.CIFAR10(root="./data", train=True, download=True)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    return PoolDataset(cifar.data, cifar.targets, transform)


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

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()


def train_active_learning(
    model: nn.Module,
    controller: ActiveLearningController,
    strategy: UncertaintySampling,
    epochs: int,
    batch_size: int,
    device: torch.device,
):
    controller.reset()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        pool_loader = controller.get_unlabeled_loader(batch_size)

        indices = strategy.select_samples(model, pool_loader, batch_size, device)
        controller.label_samples(indices)

        train_loader = controller.get_labeled_loader(batch_size)

        train_one_epoch(model, train_loader, optimizer, criterion, device)

        train_acc = evaluate(model, train_loader, device)

        print(f"[Epoch {epoch}] train accuracy: {train_acc:.4f}")


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            preds = logits.argmax(dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total


def main():
    dataset = get_dataset()
    model = get_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    controller = ActiveLearningController(dataset)
    strategy = LeastConfidenceSampling()
    train_active_learning(
        model, controller, strategy, epochs=10, batch_size=32, device=device
    )


if __name__ == "__main__":
    main()
