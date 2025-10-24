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

cd "$WORKDIR"

# Launch HPCCG with the configured grid dimensions. Allow MPI_IFACE override.
srun --mpi=${MPI_IFACE:-pmi2} --cpu-bind=cores ./test_HPCCG_mpi "$NX" "$NY" "$NZ"
