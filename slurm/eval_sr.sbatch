#!/usr/bin/env bash
#SBATCH --job-name=eval-sr
#SBATCH --partition=gpu-a40,gpu-l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval-sr_%j.out
#SBATCH --error=logs/eval-sr_%j.err

# Exemples :
#   # Éval standard diamond + NACA
#   sbatch slurm/eval_sr.sbatch results/checkpoints/run42/FAM.pkl
#
#   # Test set OOD résolution h=0.2
#   TESTSETS="diamond diamond:0.2 naca0012:0.2" sbatch slurm/eval_sr.sbatch ...
#
#   # Sans NACA
#   TESTSETS="diamond" sbatch slurm/eval_sr.sbatch results/checkpoints/run42/FAM.pkl
#
#   # Sweep limité pour test rapide
#   FULL_EVAL=1 N_EVAL=100 sbatch slurm/eval_sr.sbatch results/checkpoints/run42/FAM.pkl

set -euo pipefail

if [[ $# -eq 0 ]]; then
    echo "ERREUR : passer au moins un checkpoint en argument."
    echo "  sbatch slurm/eval_sr.sbatch results/checkpoints/<run>/FAM.pkl"
    exit 1
fi

MODELS=("$@")

DATA_ROOT="${DATA_ROOT:-data/}"
TESTSETS="${TESTSETS:-diamond naca0012}"
LR_RES="${LR_RES:-0.1}"
HR_RES="${HR_RES:-0.025}"
OUT_DIR="${OUT_DIR:-results/eval/${SLURM_JOBID}/}"
FULL_EVAL="${FULL_EVAL:-1}"
N_EVAL="${N_EVAL:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"

mkdir -p logs "${OUT_DIR}"

source .venv/bin/activate

export MPLBACKEND=Agg
export JAX_ENABLE_X64=false
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/jax_cache"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo "##############################################################"
echo "User              : ${USER}"
echo "SLURM_JOBID       : ${SLURM_JOBID}"
echo "SLURM_JOB_NODELIST: ${SLURM_JOB_NODELIST}"
echo "Modèles           : ${MODELS[*]}"
echo "data_root         : ${DATA_ROOT}"
echo "testsets          : ${TESTSETS}"
echo "lr_res / hr_res   : ${LR_RES} / ${HR_RES}"
echo "out_dir           : ${OUT_DIR}"
echo "full_eval         : ${FULL_EVAL}  (n_eval=${N_EVAL})"
echo "batch_size        : ${BATCH_SIZE}"
nvidia-smi || true
echo "##############################################################"

EXTRA_ARGS=()

if [[ "${FULL_EVAL}" == "1" ]]; then
    EXTRA_ARGS+=(--full_eval)
    if [[ "${N_EVAL}" -gt 0 ]]; then
        EXTRA_ARGS+=(--n_eval "${N_EVAL}")
    fi
fi

echo "[$(date +%T)] Démarrage évaluation SR-CFD"

python -u evaluate.py \
    --models     "${MODELS[@]}" \
    --data       "${DATA_ROOT}" \
    --testsets   ${TESTSETS}    \
    --lr_res     "${LR_RES}"    \
    --hr_res     "${HR_RES}"    \
    --out_dir    "${OUT_DIR}"   \
    --batch_size "${BATCH_SIZE}" \
    "${EXTRA_ARGS[@]}"

echo "[$(date +%T)] Done — résultats dans ${OUT_DIR}"
