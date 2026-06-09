from torch.utils.data import DataLoader, Dataset
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


class ActiveLearningController:
    def __init__(self, dataset: IndexedSubset):
        self.dataset = dataset
        self.labeled_idx: set[int] = set()
        self.unlabeled_idx: set[int] = set(range(len(dataset)))

    def label_samples(self, new_indices: list[int]):
        set_indices = set(new_indices)
        for idx in set_indices:
            self.labeled_idx.add(idx)
            self.unlabeled_idx.remove(idx)

    def get_labeled_loader(self, batch_size=32) -> DataLoader:
        return DataLoader(
            IndexedSubset(self.dataset, list(self.labeled_idx)),
            batch_size,
            shuffle=True,
        )

    def get_unlabeled_loader(self, batch_size=32) -> DataLoader:
        return DataLoader(
            IndexedSubset(self.dataset, list(self.unlabeled_idx)),
            batch_size,
            shuffle=False,
        )

    def reset(self):
        self.labeled_idx = set()
        self.unlabeled_idx = set(range(len(self.dataset)))
