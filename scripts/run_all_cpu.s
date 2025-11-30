#!/bin/bash
#
#SBATCH --job-name=medsched_cpu
#SBATCH --partition=cpu          # <-- change to your CPU partition name
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8        # adjust if needed
#SBATCH --time=02:00:00
#SBATCH --output=logs/medsched_cpu_%j.out
#SBATCH --error=logs/medsched_cpu_%j.err

echo "Starting CPU profiling job on node: $(hostname)"
date

# ====== ENV SETUP ======
# Uncomment and adapt one of these based on your cluster setup

# 1) If you use modules:
# module load anaconda/2023.09

# 2) If you use conda directly:
source ~/miniconda3/etc/profile.d/conda.sh
conda activate medsched_env       # <-- change to your env name

# ====== PROJECT ROOT ======
# cd into your project directory
cd /path/to/your/project/root     # <-- change this

mkdir -p results/profiles
mkdir -p logs

timestamp=$(date +"%Y%m%d_%H%M%S")
echo "[INFO] CPU run timestamp: $timestamp"

WORKLOADS=("W1" "W2" "W3" "W4" "W5" "W6" "W7")

for W in "${WORKLOADS[@]}"; do
  echo "----------------------------------------"
  echo "[CPU] Running workload: $W"
  echo "----------------------------------------"

  python -m src.training.train_baseline \
    --workload_id "$W" \
    --device cpu \
    --epochs 1 \
    --profile \
    --run_id "cpu_${W}_${timestamp}" \
    2>&1 | tee "logs/cpu_${W}_${timestamp}.log"

  echo "[CPU] Completed workload: $W"
  echo
done

echo "CPU profiling done."
date
