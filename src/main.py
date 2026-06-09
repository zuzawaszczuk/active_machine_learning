from random import random

import pandas as pd
from controller import ActiveLearningController
from dataset import IndexedSubset
from methods import UncertaintySampling
from methods import LeastConfidenceSampling, MarginSampling, RatioOfConfidenceSampling, EntropySampling, RandomSampling
import torch.nn as nn
import torch.optim as optim
import torch
from training_helpers import get_dataset, get_model, evaluate, train_one_epoch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


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
    "random": RandomSampling(),
}


def plot_wide(df: pd.DataFrame, name: str) -> None:
    plt.figure(figsize=(10, 6))

    for col in df.columns:
        if col == "amount_of_labeled_dataset":
            continue

        plt.plot(df["amount_of_labeled_dataset"], df[col], marker="o", label=col)

    plt.xlabel("Amount of Labeled Data")
    plt.ylabel("Validation accuracy")
    plt.title("Active Learning comparison")
    plt.legend()
    plt.grid(True)

    plt.savefig(name)
    plt.close()

def set_seed(run: int) -> None:
    torch.manual_seed(run)
    np.random.seed(run)
    random.seed(run)
    

def main():
    EPOCHS = 100
    BATCH_SIZE = 128
    N_RUNS = 1

    train_dataset, val_dataset, test_dataset = get_dataset()
    val_loader = DataLoader(IndexedSubset(val_dataset, list(range(len(val_dataset)))), batch_size=BATCH_SIZE, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.DataFrame()
    df["amount_of_labeled_dataset"] = np.arange(1, EPOCHS+1) * BATCH_SIZE

    controller = ActiveLearningController(train_dataset)
    for name, strategy in dict_methods.items():
        print(f"Training with strategy: {name}")

        all_runs = []
        for run in range(N_RUNS):
            print(f"Run {run+1}/{N_RUNS}")
            val_accs = train_active_learning(controller, strategy, EPOCHS, BATCH_SIZE, val_loader, device=device)
            all_runs.append(val_accs)
        df[name] = np.mean(all_runs, axis=0)

    name = f"comparison_cifar10_epochs_{EPOCHS}_batch_size_{BATCH_SIZE}_runs_{N_RUNS}.png"
    plot_wide(df, name=f"{name}.png")
    df.to_csv(f"results_{name}.csv", index=False)

if __name__ == "__main__":
    main()
