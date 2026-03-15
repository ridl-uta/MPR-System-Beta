#!/bin/bash
#SBATCH -J minimd
#SBATCH -p debug
#SBATCH -t 00:30:00
#SBATCH --mem-per-cpu=1G
#SBATCH -o /shared/logs/minimd-%j.out

set -euo pipefail

# Allow caller overrides.
WORKDIR=${WORKDIR:-/shared/src/miniMD/ref}
MINIMD_BIN=${MINIMD_BIN:-./miniMD_openmpi}

# Build miniMD argument vector from sbatch positional args first.
# Fallbacks: MINIMD_ARGS env string, then default input deck.
if [[ "$#" -gt 0 ]]; then
  RUN_ARGS=("$@")
elif [[ -n "${MINIMD_ARGS:-}" ]]; then
  # shellcheck disable=SC2086
  eval "set -- ${MINIMD_ARGS}"
  RUN_ARGS=("$@")
else
  RUN_ARGS=(-i in.lj.miniMD)
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

JOB_REPEAT=${JOB_REPEAT:-1}
if ! [[ "$JOB_REPEAT" =~ ^[0-9]+$ ]] || [[ "$JOB_REPEAT" -lt 1 ]]; then
  echo "[ERROR] JOB_REPEAT must be a positive integer: $JOB_REPEAT" >&2
  exit 2
fi

echo "========== miniMD Slurm Launch =========="
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
echo "MINIMD_BIN              = ${MINIMD_BIN}"
echo "MINIMD_ARGS             = ${RUN_ARGS[*]}"
echo "========================================="

cd "$WORKDIR"

if [[ ! -x "$MINIMD_BIN" ]]; then
  echo "[ERROR] miniMD binary not found/executable: $MINIMD_BIN" >&2
  exit 2
fi

# Launch miniMD.
for ((job_repeat_iter=1; job_repeat_iter<=JOB_REPEAT; job_repeat_iter++)); do
  echo "[INFO] JOB_REPEAT iteration ${job_repeat_iter}/${JOB_REPEAT}"
  srun --mpi=${MPI_IFACE:-pmi2} --cpu-bind=cores "$MINIMD_BIN" "${RUN_ARGS[@]}"
done
