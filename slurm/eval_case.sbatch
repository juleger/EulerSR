#!/usr/bin/env bash
#SBATCH --job-name=eval-case
#SBATCH --partition=gpu-a40,gpu-l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval/eval-case_%j.out
#SBATCH --error=logs/eval/eval-case_%j.err

# Évaluation "cas unique" (eval_case.py) : inférence directe + FVM live LR/HR
# pour comparer un ou plusieurs modèles sur un cas de référence déjà codé
# (utils/refs.py : REFERENCE_CASES / NACA_REFERENCE_CASES).
#
# Exemples :
#   CASE_LABEL=M09_AoA5 sbatch slurm/eval_case.sbatch results/checkpoints/DAM_sr_objnorm/DAM.pkl
#
#   GEOMETRY=naca0012 CASE_LABEL=M080_AoA2 sbatch slurm/eval_case.sbatch \
#       results/checkpoints/DAM_sr_objnorm/DAM.pkl results/checkpoints/dam_sr_tf100/DAM.pkl
#
#   NO_LIVE_FVM=1 sbatch slurm/eval_case.sbatch ...   # saute la résolution FVM live (plus rapide)

set -euo pipefail

if [[ $# -eq 0 ]]; then
    echo "ERREUR : passer au moins un checkpoint en argument."
    echo "  CASE_LABEL=<label> sbatch slurm/eval_case.sbatch results/checkpoints/<run>/DAM.pkl"
    exit 1
fi

MODELS=("$@")

DATA_ROOT="${DATA_ROOT:-data/}"
GEOMETRY="${GEOMETRY:-diamond}"
LR_RES="${LR_RES:-0.1}"
HR_RES="${HR_RES:-0.025}"
CASE_LABEL="${CASE_LABEL:-}"
MACH="${MACH:-}"
AOA="${AOA:-}"
N_REPEAT="${N_REPEAT:-5}"
OUT_DIR="${OUT_DIR:-results/eval_case/}"
NO_LIVE_FVM="${NO_LIVE_FVM:-0}"

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
echo "geometry          : ${GEOMETRY}"
echo "case_label        : ${CASE_LABEL:-<mach/aoa>}"
echo "mach / aoa        : ${MACH:-·} / ${AOA:-·}"
echo "lr_res / hr_res   : ${LR_RES} / ${HR_RES}"
echo "out_dir           : ${OUT_DIR}"
nvidia-smi || true
echo "##############################################################"

EXTRA_ARGS=()
[[ -n "${CASE_LABEL}" ]] && EXTRA_ARGS+=(--case_label "${CASE_LABEL}")
[[ -n "${MACH}" ]] && EXTRA_ARGS+=(--mach "${MACH}")
[[ -n "${AOA}"  ]] && EXTRA_ARGS+=(--aoa  "${AOA}")
[[ "${NO_LIVE_FVM}" == "1" ]] && EXTRA_ARGS+=(--no_live_fvm)

echo "[$(date +%T)] Démarrage évaluation cas unique"

python -u eval_case.py \
    --models     "${MODELS[@]}" \
    --data       "${DATA_ROOT}" \
    --geometry   "${GEOMETRY}"  \
    --lr_res     "${LR_RES}"    \
    --hr_res     "${HR_RES}"    \
    --n_repeat   "${N_REPEAT}"  \
    --out_dir    "${OUT_DIR}"   \
    "${EXTRA_ARGS[@]}"

echo "[$(date +%T)] Done — résultats dans ${OUT_DIR}"
