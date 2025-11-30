from typing import Literal, Tuple, Optional

import torchvision.transforms as T
from torch.utils.data import DataLoader

import medmnist
from medmnist import INFO

DimType = Literal["2d", "3d"]


def _infer_dim(data_flag: str, dim: Optional[DimType]) -> DimType:
    if dim is not None:
        dim_l = dim.lower()
        if dim_l not in ("2d", "3d"):
            raise ValueError(f"dim must be '2d' or '3d', got: {dim}")
        return dim_l  # type: ignore
    return "3d" if data_flag.lower().endswith("3d") else "2d"


def _default_image_size(info: dict, dim: DimType, image_size: Optional[Tuple[int, ...]]) -> Tuple[int, ...]:
    if image_size is not None:
        return image_size
    shape = info.get("shape")
    if not shape:
        raise ValueError("INFO missing shape; please provide image_size explicitly")
    if dim == "2d":
        if len(shape) < 2:
            raise ValueError(f"INFO shape too short for 2d: {shape}")
        return (shape[0], shape[1])
    # 3d
    if len(shape) < 3:
        raise ValueError(f"INFO shape too short for 3d: {shape}")
    return (shape[0], shape[1], shape[2])


def _normalize_transform(n_channels: int) -> T.Normalize:
    mean = [0.5] * n_channels
    std = [0.5] * n_channels
    return T.Normalize(mean=mean, std=std)


def get_medmnist_dataloaders(
    data_flag: str,
    batch_size: int,
    dim: Optional[DimType] = None,
    image_size: Optional[Tuple[int, ...]] = None,
    num_workers: int = 4,
    download: bool = True,
    normalize: bool = True,
    pin_memory: bool = False,
):
    """
    Build train/val/test DataLoaders for a MedMNIST dataset.

    Args:
        data_flag: MedMNIST dataset key, e.g. "pathmnist", "chestmnist", "organmnist3d".
        batch_size: Batch size for all splits.
        dim: "2d" or "3d". If omitted, inferred from dataset name (suffix "3d").
        image_size: Optional target spatial size. For 2D: (H, W). For 3D: (D, H, W).
        num_workers: Number of DataLoader workers.
        download: Whether to download data if missing.
        normalize: Apply a simple 0.5/0.5 normalization per channel.
        pin_memory: Forwarded to DataLoader (set True when using GPU).

    Returns:
        train_loader, val_loader, test_loader, num_classes, in_channels
    """
    data_flag = data_flag.lower()
    if data_flag not in INFO:
        raise ValueError(
            f"Unknown MedMNIST data_flag={data_flag}. "
            f"Available keys: {list(INFO.keys())[:10]} ..."
        )

    dim_resolved: DimType = _infer_dim(data_flag, dim)
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info["python_class"])

    resolved_size = _default_image_size(info, dim_resolved, image_size)
    n_channels = info.get("n_channels", 1)

    # -------------------------
    # Define transforms
    # -------------------------
    if dim_resolved == "2d":
        if len(resolved_size) != 2:
            raise ValueError(f"For 2D data, image_size must be (H, W), got: {resolved_size}")
        h, w = resolved_size
        transform_list = [
            T.ToTensor(),
            T.Resize((h, w)),
        ]
        if normalize:
            transform_list.append(_normalize_transform(n_channels))
        transform = T.Compose(transform_list)
    else:
        # dim_resolved == "3d"
        transform_list = [T.ToTensor()]
        if normalize:
            transform_list.append(_normalize_transform(n_channels))
        # TODO: add 3D resize/crop if you need to enforce a specific volume size.
        transform = T.Compose(transform_list)

    # -------------------------
    # Instantiate datasets
    # -------------------------
    train_dataset = DataClass(split="train", transform=transform, download=download)
    val_dataset = DataClass(split="val", transform=transform, download=download)
    test_dataset = DataClass(split="test", transform=transform, download=download)

    # -------------------------
    # Build DataLoaders
    # -------------------------
    common_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    train_loader = DataLoader(train_dataset, shuffle=True, **common_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **common_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **common_kwargs)

    # -------------------------
    # Metadata: num_classes, in_channels
    # -------------------------
    num_classes = len(info["label"])
    sample_x, _ = train_dataset[0]
    in_channels = sample_x.shape[0]

    return train_loader, val_loader, test_loader, num_classes, in_channels
