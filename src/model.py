import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


def get_model() -> nn.Module:
    # model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    # model.conv1 = nn.Conv2d(
    #     in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False
    # )
    # model.maxpool = nn.Identity()
    # model.fc = nn.Linear(model.fc.in_features, 10)
    # return SimpleCNN(num_classes=100)
    return SimpleCNN(num_classes=10)
