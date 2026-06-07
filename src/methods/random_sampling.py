from .uncertainty_sampling import UncertaintySampling
import torch.nn as nn
from torch.utils.data import DataLoader
import torch
import random


class RandomSampling(UncertaintySampling):
    def select_samples(
        self, model: nn.Module, X_pool: DataLoader, n_samples: int, device: torch.device
    ) -> list[int]:
        model.eval()

        pool_indices = []
        for _, _, idx in X_pool:
            pool_indices.extend(idx.tolist())

        selected = random.sample(pool_indices, n_samples)
        return selected
