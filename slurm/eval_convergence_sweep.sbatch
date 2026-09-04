#!/usr/bin/env bash
#SBATCH --job-name=conv-sweep
#SBATCH --partition=gpu-a40,gpu-l40,gpu-rtx6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/eval/conv-sweep_%j.out
#SBATCH --error=logs/eval/conv-sweep_%j.err

# Campagne de convergence FVM en warm-start (eval_convergence_sweep.py) sur un
# sous-ensemble Mach/AoA du testset : moyenne du nb d'itérations par modèle +
# heatmaps. Coûteux (un run FVM complet par champ initial ET par cas) : ajuste
# STRIDE_MACH/STRIDE_AOA selon le temps alloué, cf. docstring de
# eval/convergence_sweep.py. --time=08:00:00 par défaut ; augmente si tu élargis
# le sweep (MAX_CASES=0 pour la grille complète, prévoir bien plus de temps).
#
# Exemples :
#   sbatch slurm/eval_convergence_sweep.sbatch results/checkpoints/DAM_dia/DAM.pkl
#
#   GEOMETRY=diamond STRIDE_MACH=2 STRIDE_AOA=2 MAX_CASES=150 \
#       sbatch slurm/eval_convergence_sweep.sbatch \
#       results/checkpoints/DAM_dia/DAM.pkl results/checkpoints/FAM_dia/FAM.pkl

set -euo pipefail

if [[ $# -eq 0 ]]; then
    echo "ERREUR : passer au moins un checkpoint en argument."
    echo "  sbatch slurm/eval_convergence_sweep.sbatch results/checkpoints/<run>/DAM.pkl"
    exit 1
fi

MODELS=("$@")

DATA_ROOT="${DATA_ROOT:-data/}"
GEOMETRY="${GEOMETRY:-diamond}"
LR_RES="${LR_RES:-0.1}"
HR_RES="${HR_RES:-0.025}"
STRIDE_MACH="${STRIDE_MACH:-4}"
STRIDE_AOA="${STRIDE_AOA:-4}"
MAX_CASES="${MAX_CASES:-60}"
# Hystérésis du critère résidu (nb de checks consécutifs sous le seuil) et
# tolérances relatives du critère ingénieur (Cd/Cl vs cible HR). Défauts de
# eval.live_fvm si non fournis, cf. eval/convergence_case.py.
PATIENCE="${PATIENCE:-}"
CD_TOL="${CD_TOL:-}"
CL_TOL="${CL_TOL:-}"
N_STEPS="${N_STEPS:-}"
N_WALL="${N_WALL:-}"
NO_IDW="${NO_IDW:-0}"
NO_PIPELINE_TIME="${NO_PIPELINE_TIME:-0}"
OUT_DIR="${OUT_DIR:-results/convergence_sweep/}"

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
echo "stride mach / aoa : ${STRIDE_MACH} / ${STRIDE_AOA}"
echo "max_cases         : ${MAX_CASES}"
echo "n_wall            : ${N_WALL:-<complet>}"
echo "out_dir           : ${OUT_DIR}"
nvidia-smi || true
echo "##############################################################"

EXTRA_ARGS=()
[[ -n "${PATIENCE}" ]] && EXTRA_ARGS+=(--patience "${PATIENCE}")
[[ -n "${CD_TOL}" ]] && EXTRA_ARGS+=(--cd_tol "${CD_TOL}")
[[ -n "${CL_TOL}" ]] && EXTRA_ARGS+=(--cl_tol "${CL_TOL}")
[[ -n "${N_STEPS}" ]] && EXTRA_ARGS+=(--n_steps "${N_STEPS}")
[[ -n "${N_WALL}" ]] && EXTRA_ARGS+=(--n_wall "${N_WALL}")
[[ "${NO_IDW}" == "1" ]] && EXTRA_ARGS+=(--no_idw)
[[ "${NO_PIPELINE_TIME}" == "1" ]] && EXTRA_ARGS+=(--no_pipeline_time)

echo "[$(date +%T)] Démarrage campagne de convergence"

python -u eval_convergence_sweep.py \
    --models      "${MODELS[@]}"      \
    --data        "${DATA_ROOT}"      \
    --geometry    "${GEOMETRY}"       \
    --lr_res      "${LR_RES}"         \
    --hr_res      "${HR_RES}"         \
    --stride_mach "${STRIDE_MACH}"    \
    --stride_aoa  "${STRIDE_AOA}"     \
    --max_cases   "${MAX_CASES}"      \
    --out_dir     "${OUT_DIR}"        \
    "${EXTRA_ARGS[@]}"

echo "[$(date +%T)] Done — résultats dans ${OUT_DIR}"
