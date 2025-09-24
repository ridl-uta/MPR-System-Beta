#!/bin/bash
#SBATCH --job-name=demo_pack
#SBATCH --export=NONE
set -euo pipefail

printf 'Job %s running on %s\n' "${SLURM_JOB_ID:-unknown}" "${SLURM_NODELIST:-unknown}"
printf 'Total tasks: %s  Tasks per node: %s\n' "${SLURM_NTASKS:-unknown}" "${SLURM_NTASKS_PER_NODE:-unknown}"

# srun uses the task layout decided by schedule_pack_jobs.py
srun hostname
