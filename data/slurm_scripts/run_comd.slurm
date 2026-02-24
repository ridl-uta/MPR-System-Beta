#!/bin/bash
#SBATCH -J comd
#SBATCH -p debug
#SBATCH -t 00:30:00
#SBATCH --mem-per-cpu=1G
#SBATCH -o /shared/logs/comd-%j.out

set -euo pipefail

# Allow caller overrides.
WORKDIR=${WORKDIR:-/shared/src/CoMD}
COMD_BIN=${COMD_BIN:-./bin/CoMD-openmp-mpi}

# Build CoMD argument vector from sbatch positional args first.
# Fallbacks: COMD_ARGS env string, then default test problem.
if [[ "$#" -gt 0 ]]; then
  RUN_ARGS=("$@")
elif [[ -n "${COMD_ARGS:-}" ]]; then
  # shellcheck disable=SC2086
  eval "set -- ${COMD_ARGS}"
  RUN_ARGS=("$@")
else
  RUN_ARGS=(-x 40 -y 40 -z 40 -N 100)
fi

# Use the PMI2-enabled Open MPI stack and pin traffic to eno1.
export PATH=/shared/ompi-pmi2/bin:$PATH
export LD_LIBRARY_PATH=/shared/ompi-pmi2/lib:$LD_LIBRARY_PATH
export OMPI_MCA_ess=pmi
export OMPI_MCA_plm=none
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=tcp,self
export OMPI_MCA_btl_tcp_if_include=eno1

# OpenMP settings derived from Slurm cpus-per-task.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "========== CoMD Slurm Launch =========="
echo "SLURM_JOBID             = ${SLURM_JOBID}"
echo "SLURM_JOB_NODELIST      = ${SLURM_JOB_NODELIST}"
echo "SLURM_NTASKS            = ${SLURM_NTASKS}"
echo "SLURM_NTASKS_PER_NODE   = ${SLURM_NTASKS_PER_NODE}"
echo "SLURM_CPUS_PER_TASK     = ${SLURM_CPUS_PER_TASK}"
echo
echo "OMP_NUM_THREADS         = ${OMP_NUM_THREADS}"
echo "OMP_PROC_BIND           = ${OMP_PROC_BIND}"
echo "OMP_PLACES              = ${OMP_PLACES}"
echo
echo "COMD_BIN                = ${COMD_BIN}"
echo "COMD_ARGS               = ${RUN_ARGS[*]}"
echo "======================================="

cd "$WORKDIR"

if [[ ! -x "$COMD_BIN" ]]; then
  echo "[ERROR] CoMD binary not found/executable: $COMD_BIN" >&2
  exit 2
fi

# Launch CoMD.
srun --mpi=${MPI_IFACE:-pmi2} --cpu-bind=cores "$COMD_BIN" "${RUN_ARGS[@]}"
