import numpy as np
from torch.utils.data import Subset, Dataset, random_split
from torchvision import datasets, transforms
from datasets import load_dataset


def imbalance_train_dataset(train_cifar, compresion, num_classes=10, seed=42):
    rng = np.random.default_rng(seed)
    targets = np.array(train_cifar.targets)

    print("Original length of training dataset:", len(train_cifar))

    unique_classes = np.unique(targets)
    static_classes = set(range(int(0.2 * num_classes)))
    selected_indices = []

    for cls in unique_classes:
        idxs = np.flatnonzero(targets == cls)

        if len(idxs) == 0:
            continue

        if cls in static_classes:
            sampled = idxs
        else:
            sampled = rng.choice(
                idxs, size=max(1, len(idxs) // compresion), replace=False
            )

        selected_indices.append(sampled)

    selected_indices = np.concatenate(selected_indices)
    rng.shuffle(selected_indices)

    print("Length of training dataset after imbalance sampling:", len(selected_indices))

    return Subset(train_cifar, selected_indices)


def get_cifar_dataset(compresion) -> tuple[Dataset, Dataset, Dataset]:
    train_cifar = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transforms.ToTensor()
    )
    test_cifar = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transforms.ToTensor()
    )

    val_size = 5000
    test_size = len(test_cifar) - val_size

    val_cifar, test_cifar = random_split(test_cifar, [val_size, test_size])

    return (
        imbalance_train_dataset(train_cifar, compresion, num_classes=10),
        val_cifar,
        test_cifar,
    )


class EuroSATDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

        self.targets = [element["label"] for element in hf_dataset]

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        image = item["image"]
        label = item["label"]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_eurosat_dataset(compresion) -> tuple[Dataset, Dataset, Dataset]:
    transform = transforms.ToTensor()
    eurosat = load_dataset("timm/eurosat-rgb")
    print(eurosat.keys())
    print(eurosat["train"][0])

    train_dataset = EuroSATDataset(eurosat["train"], transform=transform)
    val_dataset = EuroSATDataset(eurosat["validation"], transform=transform)
    test_dataset = EuroSATDataset(eurosat["test"], transform=transform)

    return (
        imbalance_train_dataset(train_dataset, compresion, num_classes=10),
        val_dataset,
        test_dataset,
    )
