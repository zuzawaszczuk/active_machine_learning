from torchvision import datasets, transforms
from dataset import ActiveLearningController, DatasetWithoutLabels

cifar = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True
)

data = cifar.data  
print(data.shape)  
targets = cifar.targets
print(len(targets))

transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset = DatasetWithoutLabels(data.tolist(), targets, transform)

controller = ActiveLearningController(dataset)