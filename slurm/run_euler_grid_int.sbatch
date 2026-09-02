#!/usr/bin/env bash
#SBATCH --job-name=diamond
#SBATCH --partition=gpu-a40,gpu-l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=20:00:00
#SBATCH --array=0-10
#SBATCH --output=logs/euler/%x_%A_%a.out
#SBATCH --error=logs/euler/%x_%A_%a.err

# Grille Euler (Mach x AoA), AoA ENTIER de -5 à 5 (pas 1, 11 tâches) -- variante
# de run_euler_grid.sbatch. Usage : sbatch slurm/run_euler_grid_int.sbatch
# Env vars : CASE (géométrie), MESH_PATHS (1+ maillages, alias legacy MESH_PATH),
#   MACH (start:stop:step), AOA_MIN/AOA_STEP (défaut -5/1), TF (défaut 5.0),
#   STATIONARITY_THRESHOLD (défaut 1e-8)


CASE="${CASE:-diamond}"
MESH_PATHS="${MESH_PATHS:-${MESH_PATH:-data/meshes/diamond_h0.3.npy,data/meshes/diamond_h0.0175.npy,data/meshes/diamond_h0.0125.npy}}"
IFS=', ' read -r -a MESH_ARRAY <<< "${MESH_PATHS}"
MACH="${MACH:-2.05:3.00:0.05}"
TF="${TF:-2.0}"
STATIONARITY_THRESHOLD="${STATIONARITY_THRESHOLD:-1e-8}"
AOA_MIN="${AOA_MIN:--5}"
AOA_STEP="${AOA_STEP:-1}"
AOA=$(awk "BEGIN {printf \"%d\", $AOA_MIN + $SLURM_ARRAY_TASK_ID * $AOA_STEP}")

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
