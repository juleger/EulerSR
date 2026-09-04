#!/usr/bin/env bash
#SBATCH --job-name=train
#SBATCH --partition=gpu-a40,gpu-l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train/train_%j.out
#SBATCH --error=logs/train/train_%j.err

# Usage : sbatch slurm/train.sbatch <config> [run_name]   (ex: configs/fam_sr.yaml fm_v1)
# Env vars optionnelles (surchargent le YAML) : DATA_ROOT, GEOMETRY, MODEL, EPOCHS,
#   VAL_EVERY, TRAIN_FRACTION, AOA_STEP (ne garde que les AoA multiples), BATCH_SIZE,
#   INIT_FROM (checkpoint pour fine-tuning). Les env vars du shell soumetteur sont
#   propagées par `sbatch` sur ce cluster (vérifié empiriquement).

set -euo pipefail

CONFIG=${1:-configs/fam_sr.yaml}
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
echo "DATA_ROOT         : ${DATA_ROOT:-<from config>}"
echo "GEOMETRY          : ${GEOMETRY:-<from config>}"
echo "MODEL             : ${MODEL:-<from config>}"
echo "INIT_FROM         : ${INIT_FROM:-<none, init aleatoire>}"
echo "SLURM_JOB_NODELIST: ${SLURM_JOB_NODELIST}"
echo "GPU Info:"
nvidia-smi || true
echo "#############################"

echo "[$(date +%T)] Launching train.py  config=${CONFIG}"
python -u train.py \
    --config "${CONFIG}" \
    ${MODEL:+--model      "${MODEL}"} \
    ${DATA_ROOT:+--data   "${DATA_ROOT}"} \
    ${GEOMETRY:+--geometry "${GEOMETRY}"} \
    ${RUN_NAME:+--run_name "${RUN_NAME}"} \
    ${EPOCHS:+--epochs "${EPOCHS}"} \
    ${VAL_EVERY:+--val_every "${VAL_EVERY}"} \
    ${TRAIN_FRACTION:+--train_fraction "${TRAIN_FRACTION}"} \
    ${AOA_STEP:+--aoa_step "${AOA_STEP}"} \
    ${BATCH_SIZE:+--batch_size "${BATCH_SIZE}"} \
    ${INIT_FROM:+--init_from "${INIT_FROM}"}

echo "Done — $(date)"
