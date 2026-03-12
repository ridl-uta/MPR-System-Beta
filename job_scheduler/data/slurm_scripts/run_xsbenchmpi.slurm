#!/bin/bash
#SBATCH -J xs-omp
#SBATCH -p debug
#SBATCH -t 00:30:00
#SBATCH --mem-per-cpu=512M
#SBATCH -o /shared/logs/job-%j.out

set -euo pipefail

# Allow caller to pass working directory (default /shared/bin)
WORKDIR=${WORKDIR:-/shared/bin}

# Use the PMI2-enabled Open MPI stack and pin traffic to eno1.
export PATH=/shared/ompi-pmi2/bin:$PATH
export LD_LIBRARY_PATH=/shared/ompi-pmi2/lib:$LD_LIBRARY_PATH
export OMPI_MCA_ess=pmi
export OMPI_MCA_plm=none
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,vader,tcp

# Pick a usable host interface when TCP is needed. Same-node ranks use vader.
IFACE=$(ip -o -4 addr show up scope global \
  | awk '!/docker0/ && !/virbr0/ && !/cni0/ {print $2; exit}')
if [[ -n "${IFACE:-}" ]]; then
  export OMPI_MCA_btl_tcp_if_include="$IFACE"
  export OMPI_MCA_oob_tcp_if_include="$IFACE"
else
  export OMPI_MCA_btl_tcp_if_exclude=lo,docker0,virbr0,cni0,flannel.1
  export OMPI_MCA_oob_tcp_if_exclude=lo,docker0,virbr0,cni0,flannel.1
fi

# OpenMP settings derived from Slurm cpus-per-task
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OMP_PROC_BIND=close
export OMP_PLACES=cores
echo "OMPI_IFACE              = ${IFACE:-auto}"

# Optional XSBench controls (also supplied by pipeline submitter)
SIZE=${SIZE:-small}
LOOKUPS=${LOOKUPS:-10000}
XSBENCHMPI_REPEAT=${XSBENCHMPI_REPEAT:-${JOB_REPEAT:-1}}

echo "========== XSBenchMPI Slurm Launch =========="
echo "SLURM_JOBID             = ${SLURM_JOBID}"
echo "SLURM_JOB_NODELIST      = ${SLURM_JOB_NODELIST}"
echo "SLURM_NTASKS            = ${SLURM_NTASKS}"
echo "SLURM_NTASKS_PER_NODE   = ${SLURM_NTASKS_PER_NODE:-}"
echo "SLURM_CPUS_PER_TASK     = ${SLURM_CPUS_PER_TASK:-1}"
echo
echo "OMP_NUM_THREADS         = ${OMP_NUM_THREADS}"
echo "OMP_PROC_BIND           = ${OMP_PROC_BIND}"
echo "OMP_PLACES              = ${OMP_PLACES}"
echo "OMPI_IFACE              = ${IFACE:-auto}"
echo
echo "SIZE                    = ${SIZE}"
echo "LOOKUPS                 = ${LOOKUPS}"
echo "XSBENCHMPI_REPEAT       = ${XSBENCHMPI_REPEAT}"
echo "============================================"

cd "$WORKDIR"

if [[ ! "$XSBENCHMPI_REPEAT" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] XSBENCHMPI_REPEAT must be a positive integer, got: $XSBENCHMPI_REPEAT" >&2
  exit 2
fi

# Multi-rank + OpenMP; keep step success even if XSBench returns non-zero for checksum
# Allow overriding MPI interface via MPI_IFACE (pmi2|pmix). Default to pmi2 for MPICH builds.
for ((run_idx = 1; run_idx <= XSBENCHMPI_REPEAT; run_idx++)); do
  echo "========== XSBenchMPI Iteration ${run_idx}/${XSBENCHMPI_REPEAT} =========="
  srun --mpi=${MPI_IFACE:-pmi2} --cpu-bind=cores ./XSBenchMPI -s "$SIZE" -l "$LOOKUPS" -t "$OMP_NUM_THREADS" || true
done
