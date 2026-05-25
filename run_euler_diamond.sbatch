#!/usr/bin/env bash
#SBATCH --job-name=euler-diamond
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --array=0-39%8
#SBATCH --output=logs/%x/%A_%a.out
#SBATCH --error=logs/%x/%A_%a.err

set -euo pipefail

source ".venv/bin/activate"
mkdir -p "logs/${SLURM_JOB_NAME}"

module purge
module load cuda
module load uv

echo "PTXAS utilisé:"
which ptxas
ptxas --version

export MPLBACKEND=Agg
export JAX_ENABLE_X64=true
export JAX_PLATFORM_NAME=gpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

JAX_CACHE_ROOT="${SCRATCH:-$HOME}/.cache/jax"
mkdir -p "$JAX_CACHE_ROOT"
export JAX_ENABLE_COMPILATION_CACHE=true
export JAX_COMPILATION_CACHE_DIR="$JAX_CACHE_ROOT"

meshes=(
  "meshes/diamond/diamond_h0.1.npy"
  "meshes/diamond/diamond_h0.07.npy"
  "meshes/diamond/diamond_h0.05.npy"
  "meshes/diamond/diamond_h0.035.npy"
  "meshes/diamond/diamond_h0.025.npy"
  "meshes/diamond/diamond_h0.0175.npy"
  "meshes/diamond/diamond_h0.0125.npy"
  "meshes/diamond/diamond_h0.00875.npy"
)

fluxes=(Rusanov Tadmor AUSM Roe HLLC)

n_fluxes=${#fluxes[@]}  # 5

mesh_idx=$(( SLURM_ARRAY_TASK_ID / n_fluxes ))
flux_idx=$(( SLURM_ARRAY_TASK_ID % n_fluxes ))

mesh="${meshes[$mesh_idx]}"
flux="${fluxes[$flux_idx]}"

echo "#############################"
echo "User:" $USER
echo "SLURM_JOBID:" $SLURM_JOBID
echo "SLURM_SUBMIT_DIR:" $SLURM_SUBMIT_DIR
echo "SLURM_JOB_NODELIST:" $SLURM_JOB_NODELIST
echo "SLURM_ARRAY_TASK: $SLURM_ARRAY_TASK_ID : mesh=$mesh  flux=$flux"
echo "GPU Info:"
nvidia-smi
echo "#############################"

echo "[$(date +%T)] Starting flux=${flux} on mesh=${mesh}"

uv run python Euler/main.py \
  --case diamond \
  --mesh-path "$mesh" \
  --flux "$flux" \
  --time-scheme SRK2 \
  --reconstruction MUSCL
