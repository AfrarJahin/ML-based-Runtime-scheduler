import torch
import torch.nn as nn


class SimpleCNN3D(nn.Module):
    """
    Simple but stable 3D CNN for volumetric MedMNIST (e.g., OrganMNIST3D).
    Uses adaptive pooling so classifier is input-size agnostic (28^3 or 64^3).
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_dim: int = 128,
        width_mult: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        c1, c2, c3 = [k * width_mult for k in (16, 32, 64)]

        self.features = nn.Sequential(
            nn.Conv3d(in_channels, c1, kernel_size=3, padding=1),
            nn.BatchNorm3d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(c1, c2, kernel_size=3, padding=1),
            nn.BatchNorm3d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(c2, c3, kernel_size=3, padding=1),
            nn.BatchNorm3d(c3),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(c3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.head(x)


def build_cnn3d(in_channels: int, num_classes: int, **kwargs) -> nn.Module:
    """Convenience builder for config-driven instantiation."""
    return SimpleCNN3D(in_channels=in_channels, num_classes=num_classes, **kwargs)


# Backward compatibility
CNN3D = SimpleCNN3D
