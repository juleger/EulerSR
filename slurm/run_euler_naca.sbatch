#!/usr/bin/env bash
#SBATCH --job-name=euler-naca
#SBATCH --partition=gpu-a40,gpu-l40,gpu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-10
#SBATCH --output=logs/naca_%A_%a.out
#SBATCH --error=logs/naca_%A_%a.err

# 11 tâches parallèles (une par AoA de -5° à +5°)
set -euo pipefail

CASE="naca"
MESH_HR="meshes/naca/naca0012_h0.025.npy"
MESH_LR="meshes/naca/naca0012_h0.1.npy"
MACH="0.70:2.00:0.02"
AOA=$(( -5 + SLURM_ARRAY_TASK_ID ))
TF="4.0"
STATIONARITY_THRESHOLD="1e-8"

mkdir -p logs

source .venv/bin/activate

export MPLBACKEND=Agg
export JAX_ENABLE_X64=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/jax_cache"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo "#############################"
echo "User              : ${USER}"
echo "SLURM_JOBID       : ${SLURM_JOB_ID}"
echo "SLURM_ARRAY_TASK  : ${SLURM_ARRAY_TASK_ID}  (AoA=${AOA}°)"
echo "SLURM_JOB_NODELIST: ${SLURM_JOB_NODELIST}"
nvidia-smi || true
echo "#############################"

for mesh_path in "${MESH_HR}" "${MESH_LR}"; do
    if [[ ! -f "${mesh_path}" ]]; then
        echo "Maillage manquant : ${mesh_path} — générer d'abord avec : python euler/naca.py"
        exit 1
    fi
done

# Copie des maillages (force pour prendre en compte un éventuel rebuild)
mkdir -p data/naca
cp -f "${MESH_HR}" "data/naca/mesh_0.025.npy"
cp -f "${MESH_LR}" "data/naca/mesh_0.1.npy"

echo ""
echo "[$(date +%T)] === HR h=0.025 | AoA=${AOA}° | Mach ${MACH} ==="
python -u slurm/run_euler_grid.py \
    --case "${CASE}" \
    --mesh-path "${MESH_HR}" \
    --mach "${MACH}" \
    --aoa "${AOA}" \
    --tf "${TF}" \
    --stationarity-threshold "${STATIONARITY_THRESHOLD}"

echo ""
echo "[$(date +%T)] === LR h=0.1  | AoA=${AOA}° | Mach ${MACH} ==="
python -u slurm/run_euler_grid.py \
    --case "${CASE}" \
    --mesh-path "${MESH_LR}" \
    --mach "${MACH}" \
    --aoa "${AOA}" \
    --tf "${TF}" \
    --stationarity-threshold "${STATIONARITY_THRESHOLD}"

echo ""
echo "[$(date +%T)] AoA=${AOA}° terminé."
echo "Lancer le preprocessing après toutes les tâches :"
echo "  python preprocessing/preprocess.py --data_dir data/naca/"
