from controller import ActiveLearningController
from dataset import IndexedSubset
from methods import UncertaintySampling
from methods import LeastConfidenceSampling
import torch.nn as nn
import torch.optim as optim
import torch
from training_helpers import get_dataset, get_model, evaluate, train_one_epoch
from torch.utils.data import DataLoader


def train_active_learning(
    model: nn.Module,
    controller: ActiveLearningController,
    strategy: UncertaintySampling,
    epochs: int,
    batch_size: int,
    val_loader: DataLoader,
    device: torch.device,
):
    controller.reset()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        print(len(controller.labeled_idx), len(controller.unlabeled_idx))
        pool_loader = controller.get_unlabeled_loader(batch_size)

        indices = strategy.select_samples(model, pool_loader, batch_size, device)
        controller.label_samples(indices)

        train_loader = controller.get_labeled_loader(batch_size)
        train_one_epoch(model, train_loader, optimizer, criterion, device)

        train_acc = evaluate(model, train_loader, device)
        print(f"[Epoch {epoch}] train accuracy: {train_acc:.4f}")
        val_acc = evaluate(model, val_loader, device)
        print(f"[Epoch {epoch}] validation accuracy: {val_acc:.4f}")


def main():
    train_dataset, val_dataset, test_dataset = get_dataset()
    val_loader = DataLoader(IndexedSubset(val_dataset, list(range(len(val_dataset)))), batch_size=32, shuffle=False)
    model = get_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    controller = ActiveLearningController(train_dataset)
    strategy = LeastConfidenceSampling()
    train_active_learning(model, controller, strategy, 10, 32, val_loader, device=device)

if __name__ == "__main__":
    main()
