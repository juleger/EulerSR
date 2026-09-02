#!/usr/bin/env bash
#SBATCH --job-name=train-wall
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/train/train-wall_%j.out
#SBATCH --error=logs/train/train-wall_%j.err

# Usage : sbatch slurm/train_wall.sbatch <config> [run_name]
# Script SEPARE de slurm/train.sbatch (pipeline DAM/FAM/SIAM legacy) --
# appelle train_wall.py, jamais train.py, aucun couplage avec le pipeline legacy.

set -euo pipefail

CONFIG=${1:-configs/fam_wall_sd.yaml}
RUN_NAME=${2:-}

mkdir -p logs/train "${SLURM_TMPDIR:-/tmp}"

source .venv/bin/activate

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
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
nvidia-smi
echo "#############################"

python train_wall.py --config "${CONFIG}" \
    ${RUN_NAME:+--run_name "${RUN_NAME}"}
