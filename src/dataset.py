from torch.utils.data import Dataset
import torch


class IndexedSubset(Dataset):
    def __init__(self, dataset: Dataset, indices: list[int]):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int, int]:
        real_idx = self.indices[i]
        x, y = self.dataset[real_idx]
        return x, y, real_idx