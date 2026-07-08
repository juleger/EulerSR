#!/usr/bin/env bash
#SBATCH --job-name=oneraD
#SBATCH --partition=gpu-a40,gpu-l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=20:00:00
#SBATCH --array=0-20
#SBATCH --output=logs/euler/%x_%A_%a.out
#SBATCH --error=logs/euler/%x_%A_%a.err

# Grille de simulations Euler, un SLURM array par AoA, chaque tâche couvre
# une plage de Mach pour une géométrie donnée.
#
# Usage :
#   sbatch slurm/run_euler_grid.sbatch
#
# Paramètres (modifier directement ou via env vars) :
#   CASE       : géométrie  — diamond | naca0012 | rae2822 | oneraD
#   MESH_PATHS : un ou plusieurs maillages, séparés par des virgules et/ou
#                espaces (ex: "mesh_LR.npy,mesh_HR.npy" pour générer LR et HR).
#                Chaque maillage exécute la grille (Mach x AoA) complète.
#                MESH_PATH reste accepté pour la compatibilité (maillage unique).
#   MACH       : plage Mach au format start:stop:step  (ex: 0.70:4.00:0.01)
#   AOA_MIN    : AoA minimum correspondant au SLURM array task 0 (défaut: -5)
#   TF         : temps final simulation (défaut: 5.0)
#   STATIONARITY_THRESHOLD : seuil de convergence (défaut: 1e-8)


CASE="${CASE:-oneraD}"
MESH_PATHS="${MESH_PATHS:-${MESH_PATH:-data/meshes/oneraD_h0.025.npy}}"
IFS=', ' read -r -a MESH_ARRAY <<< "${MESH_PATHS}"
MACH="${MACH:-0.60:2.00:0.01}"
TF="${TF:-5.0}"
STATIONARITY_THRESHOLD="${STATIONARITY_THRESHOLD:-1e-8}"
AOA_MIN="${AOA_MIN:--5.0}"
AOA_STEP="${AOA_STEP:-0.5}"
AOA=$(awk "BEGIN {printf \"%.1f\", $AOA_MIN + $SLURM_ARRAY_TASK_ID * $AOA_STEP}")

TMPDIR="${SLURM_TMPDIR:-/tmp}"
mkdir -p "${TMPDIR}" logs/euler

echo "##############################################################"
echo "User              : ${USER}"
echo "SLURM_JOBID       : ${SLURM_JOBID}"
echo "SLURM_ARRAY_TASK  : ${SLURM_ARRAY_TASK_ID}  ->  AoA=${AOA}"
echo "Case              : ${CASE}"
echo "Meshes            : ${MESH_ARRAY[*]}"
echo "Mach range        : ${MACH}"
echo "tf / threshold    : ${TF} / ${STATIONARITY_THRESHOLD}"
echo "SLURM_JOB_NODELIST: ${SLURM_JOB_NODELIST}"
nvidia-smi || true
echo "##############################################################"

source .venv/bin/activate

export MPLBACKEND=Agg
export JAX_ENABLE_X64=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR="${TMPDIR}/jax_cache"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Mode concis par défaut (logs slurm allégés) ; VERBOSE=1 pour les logs détaillés.
VERBOSE_FLAG=""
if [ "${VERBOSE:-0}" = "1" ]; then
    VERBOSE_FLAG="--verbose"
fi

# Une exécution complète de la grille (Mach x AoA) par maillage.
for MESH_PATH in "${MESH_ARRAY[@]}"; do
    echo "[$(date +%T)] Lancement : case=${CASE}  mesh=${MESH_PATH}  AoA=${AOA}  Mach=${MACH}"

    python -u slurm/run_euler_grid.py \
        --case                   "${CASE}" \
        --mesh-path              "${MESH_PATH}" \
        --mach                   "${MACH}" \
        --aoa                    "${AOA}" \
        --tf                     "${TF}" \
        --stationarity-threshold "${STATIONARITY_THRESHOLD}" \
        ${VERBOSE_FLAG}

    echo "[$(date +%T)] Done — mesh=${MESH_PATH}  AoA=${AOA}"
done
