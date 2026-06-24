#!/usr/bin/env bash
#SBATCH --job-name=sr-train
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/sr-train_%j.out
#SBATCH --error=logs/sr-train_%j.err

# Usage:
#   sbatch train_sr.sbatch <config.yaml> [run_name]
#
# Examples:
#   sbatch slurm/train_sr.sbatch configs/crossattention_sr.yaml
#   sbatch slurm/train_sr.sbatch configs/mcanet_sr.yaml mcanet_residual_v1
#   sbatch slurm/train_sr.sbatch configs/idwmlp.yaml

set -euo pipefail

CONFIG=${1:?"Usage: sbatch train_sr.sbatch <config.yaml> [run_name]"}
RUN_NAME=${2:-}

mkdir -p logs "${SLURM_TMPDIR:-/tmp}"

source .venv/bin/activate

export MPLBACKEND=Agg
export JAX_ENABLE_X64=false
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/jax_cache"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo "#############################"
echo "User              : ${USER}"
echo "SLURM_JOBID       : ${SLURM_JOBID}"
echo "Config            : ${CONFIG}"
echo "Run name          : ${RUN_NAME:-<auto>}"
echo "SLURM_JOB_NODELIST: ${SLURM_JOB_NODELIST}"
echo "GPU Info:"
nvidia-smi || true
echo "#############################"

echo "[$(date +%T)] Launching train.py  config=${CONFIG}"
python -u train.py \
    --config   "${CONFIG}" \
    --data     data/ \
    ${RUN_NAME:+--run_name "${RUN_NAME}"}

echo "Done — $(date)"
