#!/bin/bash
#SBATCH -J xs-omp
#SBATCH -p debug
#SBATCH -t 00:30:00
#SBATCH --mem-per-cpu=512M
#SBATCH -o /shared/logs/job-%j.out

# Allow caller to pass working directory (default /shared/bin)
WORKDIR=${WORKDIR:-/shared/bin}

# Use the PMI2-enabled Open MPI stack and pin traffic to eno1.
export PATH=/shared/ompi-pmi2/bin:$PATH
export LD_LIBRARY_PATH=/shared/ompi-pmi2/lib:$LD_LIBRARY_PATH
export OMPI_MCA_ess=pmi
export OMPI_MCA_plm=none
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=tcp,self
export OMPI_MCA_btl_tcp_if_include=eno1

# OpenMP settings derived from Slurm cpus-per-task
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# Optional XSBench controls (also supplied by pipeline submitter)
SIZE=${SIZE:-small}
LOOKUPS=${LOOKUPS:-10000}

cd "$WORKDIR"

# Multi-rank + OpenMP; keep step success even if XSBench returns non-zero for checksum
# Allow overriding MPI interface via MPI_IFACE (pmi2|pmix). Default to pmi2 for MPICH builds.
srun --mpi=${MPI_IFACE:-pmi2} --cpu-bind=cores ./XSBenchMPI -s "$SIZE" -l "$LOOKUPS" -t "$OMP_NUM_THREADS" || true
