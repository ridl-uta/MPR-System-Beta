#!/bin/bash
#SBATCH -J hpcg-hybrid
#SBATCH -o /shared/logs/hpcg-%j.out
#SBATCH -N 2
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=10
#SBATCH -t 00:30:00
#SBATCH -p debug
#SBATCH --export=ALL

# Use the PMI2-enabled Open MPI from /shared.
export PATH=/shared/ompi-pmi2/bin:$PATH
export LD_LIBRARY_PATH=/shared/ompi-pmi2/lib:$LD_LIBRARY_PATH

# OpenMP settings (per MPI rank).
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# Force Open MPI to launch via Slurm PMI2.
export OMPI_MCA_ess=pmi
export OMPI_MCA_plm=none
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,vader,tcp

# Allow override to rerun from another build tree.
WORKDIR=${WORKDIR:-/shared/src/hpcg/hpcg-3.1/build_mpi_omp/bin}

# Default problem size; bump for longer validation runs.
NX=${NX:-64}
NY=${NY:-64}
NZ=${NZ:-64}
export NX NY NZ
# Example longer run:
# NX=256; NY=256; NZ=128

echo "========== HPCG Slurm Launch =========="
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
echo "NX NY NZ                = ${NX} ${NY} ${NZ}"
echo "======================================="

cd "$WORKDIR"

srun --mpi=pmi2 --cpu-bind=verbose bash -lc '
    # Auto-detect the first UP, global IPv4 interface on 192.168.2.* per node.
    IFACE=$(ip -o -4 addr show up scope global \
      | grep "192\.168\.2\." \
      | awk "!/docker0/ && !/virbr0/ && !/cni0/ {print \$2; exit}")

    # Fallback: first usable non-virtual interface.
    if [ -z "$IFACE" ]; then
      IFACE=$(ip -o -4 addr show up scope global \
        | awk "!/docker0/ && !/virbr0/ && !/cni0/ {print \$2; exit}")
    fi

    # Apply interface selection for Open MPI on this rank host.
    if [ -n "$IFACE" ]; then
      export OMPI_MCA_btl_tcp_if_include="$IFACE"
      export OMPI_MCA_oob_tcp_if_include="$IFACE"
    else
      export OMPI_MCA_btl_tcp_if_exclude=lo,docker0,virbr0,cni0,flannel.1
      export OMPI_MCA_oob_tcp_if_exclude=lo,docker0,virbr0,cni0,flannel.1
    fi

    echo "[rank $SLURM_PROCID @ $(hostname)] Using iface: $IFACE" >&2
    echo "[rank $SLURM_PROCID] OMP_NUM_THREADS=$OMP_NUM_THREADS" >&2
    echo "Cpus_allowed_list: $(grep Cpus_allowed_list /proc/self/status | cut -f2)" >&2

    exec ./xhpcg "$NX" "$NY" "$NZ"
'
