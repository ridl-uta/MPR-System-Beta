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

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OMP_PROC_BIND=close
export OMP_PLACES=cores

cd "$WORKDIR"

if ! command -v stress-ng >/dev/null 2>&1; then
  echo "[ERR] stress-ng not found in PATH" >&2
  echo "      Install it (e.g., apt-get install stress-ng) or load the module." >&2
  exit 127
fi

# Use matrix (FP-heavy) and cpu (integer) stressors; bind cores for stability.
srun --cpu-bind=cores \
  stress-ng --matrix 0 --matrix-size "$MATRIX_SIZE" \
            --cpu 0 \
            --timeout "${DURATION}s" \
            --metrics-brief || true
# nodelist/exclude can be supplied via Slurm directives or overrides, e.g.:
#   --export=ALL,TARGET_NODE=ridlserver11
# or from sbatch flags (--nodelist/--exclude)
