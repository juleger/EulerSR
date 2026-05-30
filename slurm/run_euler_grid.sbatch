#!/usr/bin/env bash
#SBATCH --job-name=euler-grid
#SBATCH --partition=gpu-l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x/%j.out
#SBATCH --error=logs/%x/%j.err

set -euo pipefail

CASE="diamond"
MESH_PATH="meshes/diamond/diamond_h0.035.npy"
MACH="0.7:4.0:0.01"
AOA="0:5:1"
FIDELITY="lf"
TF="10.0"
STATIONARITY_THRESHOLD="1e-6"
mkdir -p logs/"${SLURM_JOB_NAME}"


source .venv/bin/activate

export MPLBACKEND=Agg
export JAX_ENABLE_X64=true
export JAX_PLATFORM_NAME=gpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

JAX_CACHE_ROOT="${SLURM_TMPDIR:-/tmp}/jax-cache"
rm -rf "${JAX_CACHE_ROOT}"
mkdir -p "${JAX_CACHE_ROOT}"
export JAX_ENABLE_COMPILATION_CACHE=true
export JAX_COMPILATION_CACHE_DIR="${JAX_CACHE_ROOT}"

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
	--fidelity "${FIDELITY}" \
	--tf "${TF}" \
	--stationarity-threshold "${STATIONARITY_THRESHOLD}"