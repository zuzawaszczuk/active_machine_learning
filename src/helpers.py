import pandas as pd
import matplotlib.pyplot as plt
import random
import torch
import numpy as np
import argparse


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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Trenowanie modelu z użyciem Active Learning / Imbalance."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Liczba epok treningowych (domyślnie: 100)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Rozmiar paczki / batch size (domyślnie: 16)",
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        default=10,
        help="Liczba uruchomień pętli / eksperymentów (domyślnie: 10)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="eurosat",
        choices=["eurosat", "cifar"],
        help="Wybór zbioru danych: eurosat lub cifar (domyślnie: eurosat)",
    )
    parser.add_argument(
        "--compresion",
        type=int,
        default=10,
        help="O ile razy zostaną zmiejszone 8 z 10 klas (domyślnie: 10)",
    )

    return parser.parse_args()
