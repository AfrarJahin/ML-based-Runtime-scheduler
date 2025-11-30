# src/config/workloads.py
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class WorkloadSpec:
    workload_id: str
    dataset: str          # 'pathmnist', 'chestmnist', 'organmnist3d'
    dim: str              # '2d' or '3d'
    image_size: Tuple[int, ...]  # (H,W) for 2D, (D,H,W) for 3D
    model_name: str       # 'simple_cnn', 'resnet18_2d', 'cnn3d'
    batch_size: int
    purpose: str


WORKLOADS: Dict[str, WorkloadSpec] = {
    "W1": WorkloadSpec(
        workload_id="W1",
        dataset="pathmnist",
        dim="2d",
        image_size=(28, 28),
        model_name="simple_cnn",
        batch_size=64,
        purpose="very light workload",
    ),
    "W2": WorkloadSpec(
        workload_id="W2",
        dataset="pathmnist",
        dim="2d",
        image_size=(64, 64),
        model_name="simple_cnn",
        batch_size=64,
        purpose="light/medium",
    ),
    "W3": WorkloadSpec(
        workload_id="W3",
        dataset="pathmnist",
        dim="2d",
        image_size=(64, 64),
        model_name="resnet18_2d",
        batch_size=32,
        purpose="medium 2D",
    ),
    "W4": WorkloadSpec(
        workload_id="W4",
        dataset="chestmnist",
        dim="2d",
        image_size=(112, 112),
        model_name="resnet18_2d",
        batch_size=32,
        purpose="medium 2D",
    ),
    "W5": WorkloadSpec(
        workload_id="W5",
        dataset="chestmnist",
        dim="2d",
        image_size=(224, 224),
        model_name="resnet18_2d",
        batch_size=16,
        purpose="heavy 2D",
    ),
    "W6": WorkloadSpec(
        workload_id="W6",
        dataset="organmnist3d",
        dim="3d",
        image_size=(28, 28, 28),
        model_name="cnn3d",
        batch_size=4,
        purpose="3D baseline",
    ),
    "W7": WorkloadSpec(
        workload_id="W7",
        dataset="organmnist3d",
        dim="3d",
        image_size=(64, 64, 64),
        model_name="cnn3d",
        batch_size=2,   # you can try 4 later
        purpose="very heavy 3D",
    ),
}
