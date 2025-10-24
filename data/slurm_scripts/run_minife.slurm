#!/bin/bash
#SBATCH -J minife
#SBATCH -p debug
#SBATCH -t 00:30:00
#SBATCH --mem-per-cpu=4G
#SBATCH -o /shared/logs/miniFE_%j.out
#SBATCH -e /shared/logs/miniFE_%j.err

set -euo pipefail

# Allow caller to override working directory; default to miniFE reference build.
WORKDIR=${WORKDIR:-/shared/src/miniFE/ref/src}

# Threads per rank (optional here; submitter also exports OMP_* for record runs)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# Grid dimensions; keep NX=NY=NZ unless experimenting.
NX=${NX:-240}
NY=${NY:-240}
NZ=${NZ:-240}

cd "$WORKDIR"

# Launch miniFE with explicit grid flags. Allow MPI_IFACE override.
srun --mpi=${MPI_IFACE:-pmi2} --cpu-bind=cores ./miniFE.x --nx=$NX --ny=$NY --nz=$NZ
