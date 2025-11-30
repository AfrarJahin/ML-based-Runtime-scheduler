import torch.nn as nn
from torchvision.models import resnet18


class ResNet18_2D(nn.Module):
    """ResNet-18 adapted for MedMNIST 2D workloads."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        pretrained: bool = False,
    ):
        super().__init__()
        # No pretrained weights by default; set pretrained=True if desired
        weights_arg = "IMAGENET1K_V1" if pretrained else None
        self.model = resnet18(weights=weights_arg)

        if in_channels != 3:
            # Replace first conv to handle 1-channel inputs (e.g., ChestMNIST)
            self.model.conv1 = nn.Conv2d(
                in_channels,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )

        # Replace fully connected layer to predict num_classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


def build_resnet18_2d(in_channels: int, num_classes: int, **kwargs) -> nn.Module:
    """
    Builder for config-driven instantiation. Accepts `pretrained` in kwargs.
    """
    return ResNet18_2D(in_channels=in_channels, num_classes=num_classes, **kwargs)


# Backward compatibility alias
ResNet2D = ResNet18_2D
