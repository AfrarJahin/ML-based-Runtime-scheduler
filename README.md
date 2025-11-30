# Hybrid Scheduler — Baseline Training & Workload Profiling

This project trains baseline CNN models on MedMNIST datasets and collects batch-level workload profiles (timings, shapes, throughput, device info, etc.) to support a Hybrid CPU–GPU Scheduler & Latency Cost Model.

## Work Completed
- ✅ Dataset loading (PathMNIST, ChestMNIST, OrganMNIST3D)
- ✅ Dataset-aware loss functions (CE for multi-class, BCE for multi-label)
- ✅ Baseline models (simple CNN, ResNet18 2D, 3D CNN)
- ✅ Full training loop with timing hooks
- ✅ Rich BatchProfiler (timings, throughput, LR, loss, CUDA memory, dims)
- ✅ CSV export of all batch workloads
- ✅ Ready for analysis & cost-model training

## 1) Environment Setup
```bash
conda env create -f environment.yml
conda activate hybrid_scheduler
```

- (Optional CPU-only) `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`
- (GPU CUDA) `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`

## 2) Project Structure (key parts)
```
configs/
  base.yaml
  datasets.yaml
  models.yaml
  workloads.py          # WORKLOADS dictionary (W1–W7)
src/
  configs/
  datasets/             # dataloaders
  experiments/
  models/               # simple_cnn, resnet18_2d, cnn3d
  profiling/            # BatchProfiler
  training/             # train_baseline.py (main)
results/                # profiler CSVs (git-ignored)
scripts/                # batch runners
```
Ensure `configs/` and all `src/` subdirs have `__init__.py` (they do).

## 3) Running Training + Profiling
Run from project root:
```bash
python -m src.training.train_baseline \
  --workload_id W1 \
  --device auto \
  --epochs 1 \
  --profile
```

Key args:
- `--workload_id` : W1..W7 as defined in `configs/workloads.py`
- `--device`      : `cpu`, `cuda`, or `auto`
- `--epochs`      : number of epochs
- `--profile`     : enable batch-level profiling (writes CSVs)

Workloads encode dataset/model/batch/size:
- W1–W5: PathMNIST/ChestMNIST (2D) with SimpleCNN/ResNet18
- W6–W7: OrganMNIST3D (3D) with CNN3D

## 4) Dataset-Aware Loss (already implemented)
- Multi-class (PathMNIST, OrganMNIST3D): `CrossEntropyLoss`, `targets.long()`
- Multi-label (ChestMNIST): `BCEWithLogitsLoss`, `targets.float()`, `sigmoid` + threshold for metrics

## 5) Batch-Level Profiling (what we log)
Per batch:
- epoch, batch_idx, global_step
- batch_size, (C, H, W) or (C, D, H, W); normalized dims dim1/dim2/dim3
- device_type/index, precision, num_workers
- data/forward/backward/step time (ms) + % breakdown
- images_per_sec
- loss, lr
- CUDA memory (allocated/reserved) if on GPU
- wall_time_s (since start)

Output CSVs: `results/profiles/<run_id>_<task_type>.csv`, e.g.  
`results/profiles/run_W1_pathmnist_simple_cnn_cpu_train.csv`

## 6) Understanding the CSV
Each row = one batch. Example interpretation: “Epoch 1, batch 40, batch size 64 on CPU: forward 59 ms, backward 58 ms, step 18 ms, throughput ~3500 img/s, loss=1.73.”

## 7) Quick Analysis Example
```python
import pandas as pd

df = pd.read_csv("results/profiles/run_W1_pathmnist_simple_cnn_cpu_train.csv")
print(df.head())
print("Avg imgs/sec:", df["images_per_sec"].mean())
print("Avg forward ms:", df["forward_time_ms"].mean())
```

## 8) Next Steps
- Collect workload CSVs for W1–W7 on CPU and GPU.
- Train latency prediction models (MLP/GBT) for CPU/GPU step_time_ms.
- Integrate a hybrid scheduler (device choice CPU vs GPU, later multi-GPU).
- Compare against CPU-only, GPU-only, and heuristic baselines.
