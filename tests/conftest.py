import pytest
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from src.controller import ActiveLearningController


class FakeModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.linear = nn.Linear(3 * 32 * 32, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.linear(x)


class FakeDataset(Dataset):
    def __init__(self, size=100):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x = torch.randn(3, 32, 32)
        y = idx % 10
        return x, y, idx


@pytest.fixture
def dataset():
    return FakeDataset(size=100)


@pytest.fixture
def model():
    return FakeModel()


@pytest.fixture
def controller(dataset):
    return ActiveLearningController(dataset)
