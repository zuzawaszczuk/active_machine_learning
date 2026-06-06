import torch
from torch.utils.data import Dataset
from typing import Callable


class DatasetWithoutLabels(Dataset):
    def __init__(self, data: list[any], targets: list[int], transform: Callable):
        self.data = data
        self.targets = targets
        self.transform = transform
        self.has_label = [False] * len(data)


    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.transform(self.data[idx])
        y = torch.tensor(self.targets[idx], dtype=torch.long)
        has_label = torch.tensor(self.has_label[idx], dtype=torch.bool)

        return x, y, has_label