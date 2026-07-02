#!/usr/bin/env bash
#SBATCH --job-name=train
#SBATCH --partition=gpu-l40,gpu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# Usage:
#   sbatch slurm/train.sbatch <config> [run_name]
#   sbatch slurm/train.sbatch configs/fam_sr.yaml fm_v1
#   sbatch slurm/train.sbatch configs/fam_multi.yaml
#
# Variables d'environnement optionnelles (surchargent les défauts du config YAML) :
#   DATA_ROOT  : racine des données (ex: data/)
#   GEOMETRY   : géométrie pour single-dataset (ex: diamond)
#   MODEL      : nom du modèle      (ex: fam, dam)

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
    ${RUN_NAME:+--run_name "${RUN_NAME}"}

echo "Done — $(date)"
