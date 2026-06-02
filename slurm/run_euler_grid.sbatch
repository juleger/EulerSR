#!/usr/bin/env bash
#SBATCH --job-name=euler-grid
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x/%j.out
#SBATCH --error=logs/%x/%j.err

set -euo pipefail

CASE="diamond"
MESH_PATH="meshes/diamond/diamond_h0.1.npy"
MACH="0.7:1.0:0.01"
AOA="-5:5:1"
TF="5.0"
STATIONARITY_THRESHOLD="1e-7"
mkdir -p logs/"${SLURM_JOB_NAME}"

TMPDIR="${SLURM_TMPDIR:-/tmp}"
mkdir -p "${TMPDIR}"

echo "AOA: ${AOA}"
source .venv/bin/activate

export MPLBACKEND=Agg
export JAX_ENABLE_X64=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR="${TMPDIR}/jax_cache"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}


echo "#############################"
echo "User: ${USER}"
echo "SLURM_JOBID: ${SLURM_JOBID}"
echo "SLURM_SUBMIT_DIR: ${SLURM_SUBMIT_DIR}"
echo "SLURM_JOB_NODELIST: ${SLURM_JOB_NODELIST}"
echo "GPU Info:"
nvidia-smi || true
echo "#############################"

echo "[$(date +%T)] Launching grid runner"
python slurm/run_euler_grid.py \
	--case "${CASE}" \
	--mesh-path "${MESH_PATH}" \
	--mach "${MACH}" \
	--aoa "${AOA}" \
	--tf "${TF}" \
	--stationarity-threshold "${STATIONARITY_THRESHOLD}"