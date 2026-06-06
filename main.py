from torchvision import datasets, transforms
from dataset import DatasetWithoutLabels

cifar = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True
)

data = cifar.data  
print(data.shape)  
targets = cifar.targets
print(len(targets))

dataset = DatasetWithoutLabels(data.tolist(), targets)