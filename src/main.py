import pandas as pd
from src.controller import ActiveLearningController
from src.dataset import IndexedSubset
from src.methods import UncertaintySampling
from src.methods import LeastConfidenceSampling, MarginSampling, RatioOfConfidenceSampling, EntropySampling
import torch.nn as nn
import torch.optim as optim
import torch
from src.training_helpers import get_dataset, get_model, evaluate, train_one_epoch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt



def train_active_learning(
    controller: ActiveLearningController,
    strategy: UncertaintySampling,
    epochs: int,
    batch_size: int,
    val_loader: DataLoader,
    device: torch.device,
):
    controller.reset()
    model = get_model()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    val_accs = []

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
        val_accs.append(val_acc)
        print(f"[Epoch {epoch}] validation accuracy: {val_acc:.4f}")
    return val_accs


dict_methods = {
    "least_confidence": LeastConfidenceSampling(),
    "margin": MarginSampling(),
    "ratio_of_confidence": RatioOfConfidenceSampling(),
    "entropy": EntropySampling(),
}


def plot_wide(df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))

    for col in df.columns:
        if col == "iteration":
            continue

        plt.plot(df["iteration"], df[col], marker="o", label=col)

    plt.xlabel("Iteration")
    plt.ylabel("Validation accuracy")
    plt.title("Active Learning comparison")
    plt.legend()
    plt.grid(True)

    plt.savefig("comparison.png")
    plt.close()

def main():
    train_dataset, val_dataset, test_dataset = get_dataset()
    val_loader = DataLoader(IndexedSubset(val_dataset, list(range(len(val_dataset)))), batch_size=32, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.DataFrame()
    df["iteration"] = range(20)

    controller = ActiveLearningController(train_dataset)
    strategy = LeastConfidenceSampling()
    for name, strategy in dict_methods.items():
        print(f"Training with strategy: {name}")
        val_accs = train_active_learning(controller, strategy, 20, 32, val_loader, device=device)
        df[name] = val_accs

    plot_wide(df)
    df.to_csv("results.csv", index=False)

if __name__ == "__main__":
    main()
