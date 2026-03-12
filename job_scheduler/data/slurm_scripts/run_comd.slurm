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
  RUN_ARGS=(-x 40 -y 40 -z 40 -N 1000)
fi

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

# OpenMP settings derived from Slurm cpus-per-task.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OMP_PROC_BIND=close
export OMP_PLACES=cores
COMD_REPEAT=${COMD_REPEAT:-${JOB_REPEAT:-1}}

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
echo "OMPI_IFACE              = ${IFACE:-auto}"
echo
echo "COMD_BIN                = ${COMD_BIN}"
echo "COMD_ARGS               = ${RUN_ARGS[*]}"
echo "COMD_REPEAT             = ${COMD_REPEAT}"
echo "======================================="

cd "$WORKDIR"

if [[ ! -x "$COMD_BIN" ]]; then
  echo "[ERROR] CoMD binary not found/executable: $COMD_BIN" >&2
  exit 2
fi

if [[ ! "$COMD_REPEAT" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] COMD_REPEAT must be a positive integer, got: $COMD_REPEAT" >&2
  exit 2
fi

for ((run_idx = 1; run_idx <= COMD_REPEAT; run_idx++)); do
  echo "========== CoMD Iteration ${run_idx}/${COMD_REPEAT} =========="
  srun --mpi=${MPI_IFACE:-pmi2} --cpu-bind=cores "$COMD_BIN" "${RUN_ARGS[@]}"
done
