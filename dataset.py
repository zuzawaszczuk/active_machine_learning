import torch
from torch.utils.data import Dataset
from typing import Callable


class PoolDataset(Dataset):
    def __init__(self, data: list[any], targets: list[int], transform: Callable):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.transform(self.data[idx])
        y = torch.tensor(self.targets[idx], dtype=torch.long)
        return x, y