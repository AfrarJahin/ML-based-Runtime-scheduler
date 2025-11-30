import csv
import time
from pathlib import Path
from typing import Iterable, Optional

import torch


class BatchProfiler:
    """
    Rich batch-level profiler that writes per-batch metrics to CSV.
    Designed to feed the latency cost-model with consistent features.
    """

    def __init__(
        self,
        save_dir: Path,
        run_id: str,
        workload_id: str,
        dataset: str,
        model_name: str,
        task_type: str,
        *,
        enabled: bool = True,
        track_cuda_memory: bool = True,
        num_workers: Optional[int] = None,
        precision: Optional[str] = None,
    ) -> None:
        self.enabled = enabled
        self.save_dir = Path(save_dir)
        self.run_id = run_id
        self.workload_id = workload_id
        self.dataset = dataset
        self.model_name = model_name
        self.task_type = task_type
        self.track_cuda_memory = track_cuda_memory
        self.num_workers = num_workers
        self.precision = precision
        self.rows: list[dict] = []
        self.start_time = time.perf_counter()
        self.global_step: int = 0

    def _device_fields(self, device: torch.device) -> tuple[str, int]:
        if device.type == "cuda":
            return "cuda", device.index if device.index is not None else 0
        return device.type, -1

    def _dims_from_shape(self, images_shape: Iterable[int]) -> tuple[int, int, int, int]:
        shape = tuple(images_shape)
        if len(shape) == 4:
            # (B, C, H, W)
            _, c, h, w = shape
            d = 0
        elif len(shape) == 5:
            # (B, C, D, H, W)
            _, c, d, h, w = shape
        else:
            raise ValueError(f"Unexpected tensor shape for profiling: {shape}")

        # 2D -> (H, W, 0), 3D -> (D, H, W)
        return c, (d or h), (h if d else w), (0 if d == 0 else w)

    def _cuda_memory_mb(self, device_index: int) -> tuple[Optional[float], Optional[float]]:
        if not (self.track_cuda_memory and torch.cuda.is_available()):
            return None, None
        try:
            alloc = torch.cuda.memory_allocated(device_index) / (1024 ** 2)
            reserved = torch.cuda.memory_reserved(device_index) / (1024 ** 2)
            return float(alloc), float(reserved)
        except Exception:
            return None, None

    def log_batch(
        self,
        *,
        epoch: int,
        batch_idx: int,
        batch_size: int,
        images_shape: Iterable[int],
        device: torch.device,
        data_time_s: float,
        forward_time_s: float,
        backward_time_s: Optional[float],
        step_time_s: float,
        split: str = "train",
        loss: Optional[float] = None,
        lr: Optional[float] = None,
    ) -> None:
        if not self.enabled:
            return

        self.global_step += 1

        c, dim1, dim2, dim3 = self._dims_from_shape(images_shape)
        device_type, device_index = self._device_fields(device)

        data_ms = data_time_s * 1000.0
        fwd_ms = forward_time_s * 1000.0
        bwd_ms = (backward_time_s or 0.0) * 1000.0
        step_ms = step_time_s * 1000.0

        images_per_sec = batch_size / step_time_s if step_time_s > 0 else 0.0
        if step_ms > 0:
            data_pct = data_ms / step_ms * 100.0
            fwd_pct = fwd_ms / step_ms * 100.0
            bwd_pct = bwd_ms / step_ms * 100.0
        else:
            data_pct = fwd_pct = bwd_pct = 0.0

        cuda_mem_alloc_mb, cuda_mem_reserved_mb = self._cuda_memory_mb(device_index)

        row = {
            # Run / workload metadata
            "run_id": self.run_id,
            "workload_id": self.workload_id,
            "dataset": self.dataset,
            "model": self.model_name,
            "task_type": self.task_type,
            "split": split,

            # Progress
            "epoch": epoch,
            "batch_idx": batch_idx,
            "global_step": self.global_step,

            # Batch / tensor shape
            "batch_size": batch_size,
            "channels": c,
            "dim1": dim1,
            "dim2": dim2,
            "dim3": dim3,

            # Device info
            "device_type": device_type,
            "device_index": device_index,

            # DataLoader/precision metadata
            "num_workers": self.num_workers,
            "precision": self.precision,

            # Timing (ms)
            "data_time_ms": data_ms,
            "forward_time_ms": fwd_ms,
            "backward_time_ms": bwd_ms,
            "step_time_ms": step_ms,

            # Timing breakdown (% of step)
            "data_time_pct": data_pct,
            "forward_time_pct": fwd_pct,
            "backward_time_pct": bwd_pct,

            # Throughput
            "images_per_sec": images_per_sec,

            # Optimization state
            "loss": loss,
            "lr": lr,

            # GPU memory (MB)
            "cuda_mem_alloc_mb": cuda_mem_alloc_mb,
            "cuda_mem_reserved_mb": cuda_mem_reserved_mb,

            # Wall clock for time-series plots
            "wall_time_s": time.perf_counter() - self.start_time,
        }

        self.rows.append(row)

    def save_csv(self) -> None:
        if not (self.enabled and self.rows):
            return

        self.save_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.save_dir / f"{self.run_id}_{self.task_type}.csv"

        fieldnames = list(self.rows[0].keys())
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)

        print(f"[PROFILE] Saved batch profile to {out_path}")
