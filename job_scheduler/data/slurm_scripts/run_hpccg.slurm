#!/bin/bash
#SBATCH -J hpccg
#SBATCH -p debug
#SBATCH -t 02:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH -o /shared/logs/hpccg-%j.out

set -euo pipefail

# Allow caller to override working directory; default to HPCCG source tree.
WORKDIR=${WORKDIR:-/shared/src/HPCCG}
HPCCG_BIN=${HPCCG_BIN:-./test_HPCCG}

# Transport layer provider for libfabric-based MPI; sockets is the safe default.
export FI_PROVIDER=${FI_PROVIDER:-sockets}

# OpenMP settings derived from Slurm cpus-per-task. These only take effect if the
# selected HPCCG binary was built with OpenMP support.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# Per-rank problem dimensions; bump for longer runs.
SIZE=${SIZE:-180}
NX=${NX:-$SIZE}
NY=${NY:-$SIZE}
NZ=${NZ:-$SIZE}
HPCCG_REPEAT=${HPCCG_REPEAT:-3}

echo "========== HPCCG Slurm Launch =========="
echo "SLURM_JOBID             = ${SLURM_JOBID}"
echo "SLURM_JOB_NODELIST      = ${SLURM_JOB_NODELIST}"
echo "SLURM_NTASKS            = ${SLURM_NTASKS}"
echo "SLURM_NTASKS_PER_NODE   = ${SLURM_NTASKS_PER_NODE:-}"
echo "SLURM_CPUS_PER_TASK     = ${SLURM_CPUS_PER_TASK:-1}"
echo
echo "OMP_NUM_THREADS         = ${OMP_NUM_THREADS}"
echo "OMP_PROC_BIND           = ${OMP_PROC_BIND}"
echo "OMP_PLACES              = ${OMP_PLACES}"
echo
echo "HPCCG_BIN               = ${HPCCG_BIN}"
echo "NX NY NZ                = ${NX} ${NY} ${NZ}"
echo "HPCCG_REPEAT            = ${HPCCG_REPEAT}"
echo "========================================"

cd "$WORKDIR"

if [[ ! -x "$HPCCG_BIN" ]]; then
  echo "[ERROR] HPCCG binary not found/executable: $HPCCG_BIN" >&2
  exit 2
fi

if [[ ! "$HPCCG_REPEAT" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] HPCCG_REPEAT must be a positive integer, got: $HPCCG_REPEAT" >&2
  exit 2
fi

# Launch HPCCG repeatedly with the configured grid dimensions. This is useful
# for extending steady-state runtime without forcing larger problem sizes.
for ((run_idx = 1; run_idx <= HPCCG_REPEAT; run_idx++)); do
  echo "========== HPCCG Iteration ${run_idx}/${HPCCG_REPEAT} =========="
  srun --mpi=${MPI_IFACE:-pmi2} --cpu-bind=cores "$HPCCG_BIN" "$NX" "$NY" "$NZ"
done
