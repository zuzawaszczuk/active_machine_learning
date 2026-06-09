import pandas as pd
from controller import ActiveLearningController, IndexedSubset
from methods import UncertaintySampling
from methods import (
    LeastConfidenceSampling,
    MarginSampling,
    RatioOfConfidenceSampling,
    EntropySampling,
    RandomSampling,
)
import torch.nn as nn
import torch.optim as optim
import torch
from training import evaluate, train_one_epoch
from model import get_model
from dataset import get_cifar_dataset, get_eurosat_dataset
from helpers import plot_wide, set_seed, parse_args
from torch.utils.data import DataLoader
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


def main():
    args = parse_args()

    if args.dataset == "eurosat":
        train_dataset, val_dataset, test_dataset = get_eurosat_dataset(args.compresion)
    else:
        train_dataset, val_dataset, test_dataset = get_cifar_dataset(args.compresion)

    val_loader = DataLoader(
        IndexedSubset(val_dataset, list(range(len(val_dataset)))),
        batch_size=args.batch_size,
        shuffle=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.DataFrame()
    df["amount_of_labeled_dataset"] = np.arange(1, args.epochs + 1) * args.batch_size

    controller = ActiveLearningController(train_dataset)
    for name, strategy in dict_methods.items():
        print(f"Training with strategy: {name}")

        all_runs = []
        for run in range(args.n_runs):
            set_seed(run)
            print(f"Run {run+1}/{args.n_runs}")
            val_accs = train_active_learning(
                controller,
                strategy,
                args.epochs,
                args.batch_size,
                val_loader,
                device=device,
            )
            all_runs.append(val_accs)
        df[name] = np.mean(all_runs, axis=0)

    name = f"comparison_{args.dataset}_epochs_{args.epochs}_batch_size_{args.batch_size}_runs_{args.n_runs}_compresion_{args.compresion}.png"
    plot_wide(df, name=f"{name}.png")
    df.to_csv(f"results_{name}.csv", index=False)


if __name__ == "__main__":
    main()
