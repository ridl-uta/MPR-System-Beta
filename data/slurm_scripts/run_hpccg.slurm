#!/bin/bash
#SBATCH -J hpccg
#SBATCH -p debug
#SBATCH -t 02:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH -o /shared/logs/hpccg-%j.out

# Allow caller to override working directory; default to HPCCG source tree.
WORKDIR=${WORKDIR:-/shared/src/HPCCG}

# Transport layer provider for libfabric-based MPI; sockets is the safe default.
export FI_PROVIDER=${FI_PROVIDER:-sockets}

# Open MPI transport defaults for this cluster. Use the cluster subnet instead
# of a device name because interface names differ across compute nodes.
export OMPI_MCA_pml=${OMPI_MCA_pml:-ob1}
export OMPI_MCA_btl=${OMPI_MCA_btl:-self,vader,tcp}
export OMPI_MCA_btl_tcp_if_include=${OMPI_MCA_btl_tcp_if_include:-192.168.2.0/24}
export OMPI_MCA_oob_tcp_if_include=${OMPI_MCA_oob_tcp_if_include:-192.168.2.0/24}

# Deterministic OpenMP placement inside each Slurm rank allocation.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}
export OMP_PLACES=${OMP_PLACES:-cores}
export OMP_PROC_BIND=${OMP_PROC_BIND:-close}

# Per-rank problem dimensions; bump for longer runs.
SIZE=${SIZE:-180}
NX=${NX:-$SIZE}
NY=${NY:-$SIZE}
NZ=${NZ:-$SIZE}

# Optional repetition count for performance recording sweeps.
# HPCCG_REPEAT is benchmark-specific; JOB_REPEAT is the generic fallback.
JOB_REPEAT=${HPCCG_REPEAT:-${JOB_REPEAT:-1}}
if ! [[ "$JOB_REPEAT" =~ ^[0-9]+$ ]] || [[ "$JOB_REPEAT" -lt 1 ]]; then
  echo "[ERROR] JOB_REPEAT must be a positive integer: $JOB_REPEAT" >&2
  exit 2
fi

# Use contiguous rank masks for the 4-rank/2-node HPCCG recording layout.
# This avoids Slurm's default even/odd interleaving across each 20-core node.
if [[ -z "${HPCCG_CPU_BIND:-}" ]]; then
  if [[ "${SLURM_NTASKS:-}" == "4" \
      && "${SLURM_NTASKS_PER_NODE:-}" == "2" \
      && "${SLURM_CPUS_PER_TASK:-}" == "10" ]]; then
    HPCCG_CPU_BIND="mask_cpu:0x0003ff,0xffc00,0x0003ff,0xffc00"
  else
    HPCCG_CPU_BIND="cores"
  fi
fi
HPCCG_CPU_DISTRIBUTION=${HPCCG_CPU_DISTRIBUTION:-block:block}

cd "$WORKDIR"

echo "========== HPCCG Slurm Launch =========="
echo "SLURM_JOBID             = ${SLURM_JOBID:-}"
echo "SLURM_JOB_NODELIST      = ${SLURM_JOB_NODELIST:-}"
echo "SLURM_NTASKS            = ${SLURM_NTASKS:-}"
echo "SLURM_NTASKS_PER_NODE   = ${SLURM_NTASKS_PER_NODE:-}"
echo "SLURM_CPUS_PER_TASK     = ${SLURM_CPUS_PER_TASK:-1}"
echo
echo "MPI_IFACE               = pmi2"
echo "OMPI_MCA_pml            = ${OMPI_MCA_pml}"
echo "OMPI_MCA_btl            = ${OMPI_MCA_btl}"
echo "OMPI_MCA_btl_tcp_if_include = ${OMPI_MCA_btl_tcp_if_include}"
echo "OMPI_MCA_oob_tcp_if_include = ${OMPI_MCA_oob_tcp_if_include}"
echo "OMP_NUM_THREADS         = ${OMP_NUM_THREADS}"
echo "OMP_PLACES              = ${OMP_PLACES}"
echo "OMP_PROC_BIND           = ${OMP_PROC_BIND}"
echo "HPCCG_CPU_BIND          = ${HPCCG_CPU_BIND}"
echo "HPCCG_CPU_DISTRIBUTION  = ${HPCCG_CPU_DISTRIBUTION}"
echo "NX NY NZ                = ${NX} ${NY} ${NZ}"
echo "JOB_REPEAT              = ${JOB_REPEAT}"
echo "========================================"

for ((job_repeat_iter=1; job_repeat_iter<=JOB_REPEAT; job_repeat_iter++)); do
  echo "[INFO] JOB_REPEAT iteration ${job_repeat_iter}/${JOB_REPEAT}"
  srun \
    --mpi=pmi2 \
    --cpu-bind="${HPCCG_CPU_BIND}" \
    --distribution="${HPCCG_CPU_DISTRIBUTION}" \
    --hint=nomultithread \
    ./test_HPCCG "$NX" "$NY" "$NZ"
  status=$?
  if [[ "$status" -ne 0 ]]; then
    exit "$status"
  fi
done
