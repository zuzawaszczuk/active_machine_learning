import torch
from torch.utils.data import Dataset, Subset
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
    

class ActiveLearningController:
    def __init__(self, dataset: PoolDataset):
        self.dataset = dataset
        self.labeled_idx = set()
        self.unlabeled_idx = set(range(len(dataset)))

    def label_samples(self, new_indices: list[int]):
        for idx in new_indices:
            self.labeled_idx.add(idx)
            self.unlabeled_idx.remove(idx)

    def get_labeled_data(self) -> Subset:
        X_train = Subset(self.dataset, list(self.labeled_idx))
        return X_train

    def get_unlabeled_data(self) -> Subset:
        X_pool = Subset(self.dataset, list(self.unlabeled_idx))
        return X_pool