from abc import ABC, abstractmethod
import torch
from torch.utils.data import DataLoader
import torch.nn as nn


class UncertaintySampling(ABC):
    @abstractmethod
    def select_samples(self, model: nn.Module, X_pool: DataLoader, n_samples: int, device: torch.device) -> list[int]:
        pass