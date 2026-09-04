#!/usr/bin/env bash
#SBATCH --job-name=eval-sr
#SBATCH --partition=gpu-a40,gpu-l40,gpu-rtx6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval/eval-sr_%j.out
#SBATCH --error=logs/eval/eval-sr_%j.err

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
#
#   # Exclure les cas Mach>2 (tf solveur raccourci, cf. logs/euler/*)
#   MACH_MAX=2.0 sbatch slurm/eval_sr.sbatch results/checkpoints/run42/FAM.pkl
#
#   # Mode apparié : modele[i] evalue UNIQUEMENT sur testset[i] (1:1), un modele
#   # par checkpoint passe en argument, dans le meme ordre que TESTSETS
#   PAIRED=1 TESTSETS="diamond diamond_half diamond_quarter diamond_nw10" \
#     sbatch slurm/eval_sr.sbatch results/checkpoints/FAMWALL_N84/FAMWall.pkl \
#     results/checkpoints/FAMWALL_N42/FAMWall.pkl results/checkpoints/FAMWALL_N21/FAMWall.pkl \
#     results/checkpoints/FAMWALL_N10/FAMWall.pkl

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
MACH_MAX="${MACH_MAX:-}"
REF_CASES="${REF_CASES:-}"
OUT_DIR="${OUT_DIR:-results/eval/${SLURM_JOBID}/}"
OUT_DIRS="${OUT_DIRS:-}"
FULL_EVAL="${FULL_EVAL:-1}"
N_EVAL="${N_EVAL:-0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
N_STEPS="${N_STEPS:-}"
N_SAMPLES="${N_SAMPLES:-}"
CFG_SCALE="${CFG_SCALE:-}"
GEOM_CFG_SCALE="${GEOM_CFG_SCALE:-}"
GEOM_CFG_SCALE_SWEEP="${GEOM_CFG_SCALE_SWEEP:-}"
CFG_SCALE_SWEEP="${CFG_SCALE_SWEEP:-}"
N_STEPS_SWEEP="${N_STEPS_SWEEP:-}"
N_SAMPLES_SWEEP="${N_SAMPLES_SWEEP:-}"
PAIRED="${PAIRED:-0}"

mkdir -p logs "${OUT_DIR}"
if [[ -n "${OUT_DIRS}" ]]; then
    for d in ${OUT_DIRS}; do mkdir -p "${d}"; done
fi

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
echo "mach_max          : ${MACH_MAX:-<aucun>}"
echo "out_dir           : ${OUT_DIR}"
echo "full_eval         : ${FULL_EVAL}  (n_eval=${N_EVAL})"
echo "batch_size        : ${BATCH_SIZE}"
echo "paired            : ${PAIRED}"
nvidia-smi || true
echo "##############################################################"

EXTRA_ARGS=()

if [[ "${FULL_EVAL}" == "1" ]]; then
    EXTRA_ARGS+=(--full_eval)
    if [[ "${N_EVAL}" -gt 0 ]]; then
        EXTRA_ARGS+=(--n_eval "${N_EVAL}")
    fi
fi
[[ -n "${MACH_MAX}"  ]] && EXTRA_ARGS+=(--mach_max  "${MACH_MAX}")
[[ -n "${REF_CASES}" ]] && EXTRA_ARGS+=(--ref_cases ${REF_CASES})
[[ -n "${N_STEPS}"   ]] && EXTRA_ARGS+=(--n_steps   "${N_STEPS}")
[[ -n "${N_SAMPLES}" ]] && EXTRA_ARGS+=(--n_samples "${N_SAMPLES}")
[[ -n "${CFG_SCALE}" ]] && EXTRA_ARGS+=(--cfg_scale "${CFG_SCALE}")
[[ -n "${GEOM_CFG_SCALE}" ]] && EXTRA_ARGS+=(--geom_cfg_scale "${GEOM_CFG_SCALE}")
[[ -n "${GEOM_CFG_SCALE_SWEEP}" ]] && EXTRA_ARGS+=(--geom_cfg_scale_sweep ${GEOM_CFG_SCALE_SWEEP})
[[ -n "${N_STEPS_SWEEP}"   ]] && EXTRA_ARGS+=(--n_steps_sweep   ${N_STEPS_SWEEP})
[[ -n "${N_SAMPLES_SWEEP}" ]] && EXTRA_ARGS+=(--n_samples_sweep ${N_SAMPLES_SWEEP})
[[ -n "${CFG_SCALE_SWEEP}" ]] && EXTRA_ARGS+=(--cfg_scale_sweep ${CFG_SCALE_SWEEP})
[[ "${PAIRED}" == "1" ]] && EXTRA_ARGS+=(--paired)

echo "[$(date +%T)] Démarrage évaluation SR-CFD"

[[ -n "${OUT_DIRS}" ]] && EXTRA_ARGS+=(--out_dirs ${OUT_DIRS})

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
