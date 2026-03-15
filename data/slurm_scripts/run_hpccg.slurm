#!/bin/bash
#SBATCH -J hpccg
#SBATCH -p debug
#SBATCH -t 02:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH -o /shared/logs/hpccg-%j.out

# Allow caller to override working directory; default to HPCCG source tree.
WORKDIR=${WORKDIR:-/shared/src/HPCCG}

# Transport layer provider for libfabric-based MPI; sockets is the safe default.
export FI_PROVIDER=${FI_PROVIDER:-sockets}

# Per-rank problem dimensions; bump for longer runs.
SIZE=${SIZE:-180}
NX=${NX:-$SIZE}
NY=${NY:-$SIZE}
NZ=${NZ:-$SIZE}

# Optional repetition count for performance recording sweeps.
JOB_REPEAT=${JOB_REPEAT:-1}
if ! [[ "$JOB_REPEAT" =~ ^[0-9]+$ ]] || [[ "$JOB_REPEAT" -lt 1 ]]; then
  echo "[ERROR] JOB_REPEAT must be a positive integer: $JOB_REPEAT" >&2
  exit 2
fi

cd "$WORKDIR"

# Launch HPCCG with the configured grid dimensions. Allow MPI_IFACE override.
for ((job_repeat_iter=1; job_repeat_iter<=JOB_REPEAT; job_repeat_iter++)); do
  echo "[INFO] JOB_REPEAT iteration ${job_repeat_iter}/${JOB_REPEAT}"
  srun --mpi=${MPI_IFACE:-pmi2} --cpu-bind=cores ./test_HPCCG "$NX" "$NY" "$NZ"
  status=$?
  if [[ "$status" -ne 0 ]]; then
    exit "$status"
  fi
done
