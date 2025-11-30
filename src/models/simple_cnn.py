import torch
import torch.nn as nn


class SimpleCNN2D(nn.Module):
    """
    Lightweight 2D CNN for small medical images (PathMNIST/ChestMNIST).
    Uses adaptive pooling so the classifier stays stable for varied input sizes
    (e.g., 64x64, 112x112, 224x224).
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Spatially flatten to a fixed 1x1 via adaptive pooling
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.head(x)


def build_simple_cnn(in_channels: int, num_classes: int, **kwargs) -> nn.Module:
    """
    Convenience builder for config-driven instantiation.
    **kwargs are forwarded (hidden_dim, dropout).
    """
    return SimpleCNN2D(in_channels=in_channels, num_classes=num_classes, **kwargs)


# Backward compatibility with earlier imports
SimpleCNN = SimpleCNN2D
