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

# Use the PMI2-enabled Open MPI stack and route MPI TCP over eno1.
export PATH=/shared/ompi-pmi2/bin:$PATH
export LD_LIBRARY_PATH=/shared/ompi-pmi2/lib:$LD_LIBRARY_PATH
export OMPI_MCA_ess=pmi
export OMPI_MCA_plm=none
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=tcp,self
export OMPI_MCA_btl_tcp_if_include=eno1

# Threads per rank (optional here; submitter also exports OMP_* for record runs)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# Grid dimensions; keep NX=NY=NZ unless experimenting.
NX=${NX:-240}
NY=${NY:-240}
NZ=${NZ:-240}
export NX NY NZ

echo "========== miniFE Slurm Launch =========="
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
echo "========================================="

cd "$WORKDIR"

# Launch miniFE with explicit grid flags. Allow MPI_IFACE override.
srun --mpi=${MPI_IFACE:-pmi2} --cpu-bind=cores ./miniFE.x --nx=$NX --ny=$NY --nz=$NZ
