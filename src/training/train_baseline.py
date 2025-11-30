# src/training/train_baseline.py
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from configs.workloads import WORKLOADS
from src.datasets.medmnist_loader import get_medmnist_dataloaders
from src.models.simple_cnn import build_simple_cnn
from src.models.resnet_2d import build_resnet18_2d
from src.models.cnn3d import build_cnn3d
from src.profiling.profiler import BatchProfiler


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def build_model(model_name: str, in_channels: int, num_classes: int):
    if model_name == "simple_cnn":
        return build_simple_cnn(in_channels, num_classes)
    if model_name == "resnet18_2d":
        return build_resnet18_2d(in_channels, num_classes)
    if model_name == "cnn3d":
        return build_cnn3d(in_channels, num_classes)
    raise ValueError(f"Unknown model_name={model_name}")


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    epoch: int,
    device: torch.device,
    workload_id: str,
    dataset_name: str,
    model_name: str,
    run_id: str,
    profile: bool,
    num_workers: int,
    precision: str,
    multilabel: bool,
):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    epoch_start = time.perf_counter()

    profiler = BatchProfiler(
        save_dir=Path("results/profiles"),
        run_id=run_id,
        workload_id=workload_id,
        dataset=dataset_name,
        model_name=model_name,
        task_type="train",
        enabled=profile,
        num_workers=num_workers,
        precision=precision,
    )

    for batch_idx, (images, labels) in enumerate(loader):
        step_start = time.perf_counter()

        t0 = time.perf_counter()
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        labels = labels.squeeze()
        if multilabel:
            labels = labels.float()
        else:
            labels = labels.long()
        t1 = time.perf_counter()
        data_time = t1 - t0

        t2 = time.perf_counter()
        outputs = model(images)
        loss = criterion(outputs, labels)
        t3 = time.perf_counter()
        fwd_time = t3 - t2

        t4 = time.perf_counter()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        t5 = time.perf_counter()
        bwd_time = t5 - t4

        step_time = t5 - step_start

        running_loss += loss.item() * images.size(0)
        if multilabel:
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.numel()
        else:
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

        profiler.log_batch(
            epoch=epoch,
            batch_idx=batch_idx,
            batch_size=images.size(0),
            images_shape=tuple(images.shape),
            device=device,
            data_time_s=data_time,
            forward_time_s=fwd_time,
            backward_time_s=bwd_time,
            step_time_s=step_time,
            split="train",
            loss=loss.item(),
            lr=optimizer.param_groups[0].get("lr"),
        )

    profiler.save_csv()
    epoch_end = time.perf_counter()
    return running_loss / total, correct / total, epoch_end - epoch_start


@torch.no_grad()
def evaluate(model, loader, criterion, device, name="Val", profiler: BatchProfiler | None = None):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    t0 = time.perf_counter()

    multilabel = isinstance(criterion, nn.BCEWithLogitsLoss)

    for batch_idx, (images, labels) in enumerate(loader):
        step_start = time.perf_counter()
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        labels = labels.squeeze()
        if multilabel:
            labels = labels.float()
        else:
            labels = labels.long()
        t1 = time.perf_counter()

        outputs = model(images)
        loss = criterion(outputs, labels)
        t2 = time.perf_counter()

        running_loss += loss.item() * images.size(0)
        if multilabel:
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.numel()
        else:
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

        step_time = t2 - step_start

        if profiler:
            profiler.log_batch(
                epoch=-1,
                batch_idx=batch_idx,
                batch_size=images.size(0),
                images_shape=tuple(images.shape),
                device=device,
                data_time_s=t1 - step_start,
                forward_time_s=t2 - t1,
                backward_time_s=None,
                step_time_s=step_time,
                split=name.lower(),
                loss=loss.item(),
                lr=None,
            )

    if profiler:
        profiler.save_csv()

    t1 = time.perf_counter()
    print(f"{name}: loss={running_loss/total:.4f}, acc={correct/total:.4f}, time={t1-t0:.2f}s")
    return running_loss / total, correct / total, t1 - t0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload_id", type=str, required=True, help="W1..W7")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--profile", action="store_true", help="Enable batch-level profiling")
    args = parser.parse_args()

    if args.workload_id not in WORKLOADS:
        raise ValueError(f"Unknown workload_id={args.workload_id}. Choose from {list(WORKLOADS.keys())}")

    spec = WORKLOADS[args.workload_id]
    device = resolve_device(args.device)
    device_desc = str(device)
    num_workers = 4
    precision = "fp32"

    print(f"[INFO] Workload {spec.workload_id}: {spec.dataset}, {spec.image_size}, {spec.model_name}, batch={spec.batch_size}")
    print(f"[INFO] Using device: {device_desc}")

    train_loader, val_loader, test_loader, n_classes, in_channels = get_medmnist_dataloaders(
        data_flag=spec.dataset,
        dim=spec.dim,
        image_size=spec.image_size,
        batch_size=spec.batch_size,
        num_workers=num_workers,
    )

    model = build_model(spec.model_name, in_channels=in_channels, num_classes=n_classes).to(device)
    multilabel = spec.dataset == "chestmnist"
    criterion = nn.BCEWithLogitsLoss() if multilabel else nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    run_id = args.run_id or f"run_{spec.workload_id}_{spec.dataset}_{spec.model_name}_{device.type}"

    # Profilers for eval (reuse per phase)
    val_profiler = BatchProfiler(
        save_dir=Path("results/profiles"),
        run_id=run_id,
        workload_id=spec.workload_id,
        dataset=spec.dataset,
        model_name=spec.model_name,
        task_type="eval",
        enabled=args.profile,
        num_workers=num_workers,
        precision=precision,
    ) if args.profile else None

    test_profiler = BatchProfiler(
        save_dir=Path("results/profiles"),
        run_id=run_id,
        workload_id=spec.workload_id,
        dataset=spec.dataset,
        model_name=spec.model_name,
        task_type="test",
        enabled=args.profile,
        num_workers=num_workers,
        precision=precision,
    ) if args.profile else None

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, t_train = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            epoch,
            device,
            spec.workload_id,
            spec.dataset,
            spec.model_name,
            run_id,
            args.profile,
            num_workers,
            precision,
            multilabel,
        )
        print(f"Epoch {epoch}: train loss={train_loss:.4f}, acc={train_acc:.4f}, time={t_train:.2f}s")
        evaluate(model, val_loader, criterion, device, name="Val", profiler=val_profiler)

    print("Final Test:")
    evaluate(model, test_loader, criterion, device, name="Test", profiler=test_profiler)


if __name__ == "__main__":
    main()
