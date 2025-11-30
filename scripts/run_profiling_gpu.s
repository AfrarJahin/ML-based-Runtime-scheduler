#!/bin/bash
#
#SBATCH --job-name=medsched_gpu
#SBATCH --partition=gpu           # <-- change to your GPU partition name
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1              # 1 GPU; change if you want more
#SBATCH --time=04:00:00
#SBATCH --output=logs/medsched_gpu_%j.out
#SBATCH --error=logs/medsched_gpu_%j.err

echo "Starting GPU profiling job on node: $(hostname)"
nvidia-smi || echo "nvidia-smi not available"
date

# ====== ENV SETUP ======
# 1) If you use modules:
# module load anaconda/2023.09
# module load cuda/12.1

# 2) If you use conda directly:
source ~/miniconda3/etc/profile.d/conda.sh
conda activate medsched_env       # <-- change to your env name (with GPU PyTorch)

# ====== PROJECT ROOT ======
cd /path/to/your/project/root     # <-- change this

mkdir -p results/profiles
mkdir -p logs

timestamp=$(date +"%Y%m%d_%H%M%S")
echo "[INFO] GPU run timestamp: $timestamp"

WORKLOADS=("W1" "W2" "W3" "W4" "W5" "W6" "W7")

for W in "${WORKLOADS[@]}"; do
  echo "----------------------------------------"
  echo "[GPU] Running workload: $W"
  echo "----------------------------------------"

  python -m src.training.train_baseline \
    --workload_id "$W" \
    --device cuda \
    --epochs 2 \
    --profile \
    --run_id "gpu_${W}_${timestamp}" \
    2>&1 | tee "logs/gpu_${W}_${timestamp}.log"

  echo "[GPU] Completed workload: $W"
  echo
done

echo "GPU profiling done."
date
