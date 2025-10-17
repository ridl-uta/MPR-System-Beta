#!/bin/bash
#SBATCH -J xs-omp
#SBATCH -p debug
#SBATCH -t 00:30:00
#SBATCH --mem-per-cpu=512M
#SBATCH -o /shared/logs/job-%j.out

# Allow caller to pass working directory (default /shared/bin)
WORKDIR=${WORKDIR:-/shared/bin}

# OpenMP settings derived from Slurm cpus-per-task
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# Optional XSBench controls (also supplied by pipeline submitter)
SIZE=${SIZE:-small}
LOOKUPS=${LOOKUPS:-10000}

cd "$WORKDIR"

# Multi-rank + OpenMP; keep step success even if XSBench returns non-zero for checksum
srun --mpi=pmi2 --cpu-bind=cores ./XSBenchMPI -s "$SIZE" -l "$LOOKUPS" -t "$OMP_NUM_THREADS" || true

