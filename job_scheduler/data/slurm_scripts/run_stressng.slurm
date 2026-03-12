#!/bin/bash
#SBATCH -J stressng
#SBATCH -p debug
#SBATCH -t 00:15:00
#SBATCH --mem-per-cpu=512M
#SBATCH --nodelist=${TARGET_NODE:-}
#SBATCH --exclude=${EXCLUDE_NODE:-}
#SBATCH -o /shared/logs/stressng-%j.out

# Simple single-node CPU + FPU stress that responds to frequency changes.
#
# Tunables via environment (override at submit time with --export=ALL,VAR=...):
#   DURATION     seconds to run (default 180)
#   MATRIX_SIZE  size for --matrix stressor (default 2048)
#   WORKDIR      working directory (default /shared)
#
# Example:
#   sbatch -N 1 -n 1 -c 10 -t 00:10:00 \
#     --export=ALL,DURATION=180,MATRIX_SIZE=2048,WORKDIR=/shared \
#     data/slurm_scripts/run_stressng.slurm

set -euo pipefail

DURATION=${DURATION:-180}
MATRIX_SIZE=${MATRIX_SIZE:-2048}
WORKDIR=${WORKDIR:-/shared}
STRESSNG_REPEAT=${STRESSNG_REPEAT:-${JOB_REPEAT:-1}}

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "========== stress-ng Slurm Launch =========="
echo "SLURM_JOBID             = ${SLURM_JOBID}"
echo "SLURM_JOB_NODELIST      = ${SLURM_JOB_NODELIST}"
echo "SLURM_NTASKS            = ${SLURM_NTASKS}"
echo "SLURM_CPUS_PER_TASK     = ${SLURM_CPUS_PER_TASK:-1}"
echo
echo "DURATION                = ${DURATION}"
echo "MATRIX_SIZE             = ${MATRIX_SIZE}"
echo "STRESSNG_REPEAT         = ${STRESSNG_REPEAT}"
echo "==========================================="

cd "$WORKDIR"

if ! command -v stress-ng >/dev/null 2>&1; then
  echo "[ERR] stress-ng not found in PATH" >&2
  echo "      Install it (e.g., apt-get install stress-ng) or load the module." >&2
  exit 127
fi

if [[ ! "$STRESSNG_REPEAT" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] STRESSNG_REPEAT must be a positive integer, got: $STRESSNG_REPEAT" >&2
  exit 2
fi

for ((run_idx = 1; run_idx <= STRESSNG_REPEAT; run_idx++)); do
  echo "========== stress-ng Iteration ${run_idx}/${STRESSNG_REPEAT} =========="
  srun --cpu-bind=cores \
    stress-ng --matrix 0 --matrix-size "$MATRIX_SIZE" \
              --cpu 0 \
              --timeout "${DURATION}s" \
              --metrics-brief || true
done
# nodelist/exclude can be supplied via Slurm directives or overrides, e.g.:
#   --export=ALL,TARGET_NODE=ridlserver11
# or from sbatch flags (--nodelist/--exclude)
