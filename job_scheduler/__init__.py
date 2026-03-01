"""Job scheduler package.

Public API:
    - SlurmHelperUtility
    - JobScheduler
"""

from pathlib import Path

from .helper_utility import SlurmHelperUtility
from .performance_data import REQUIRED_PERF_COLUMNS, load_perf_data_for_jobs
from .scheduler import JobScheduler

MODULE_ROOT = Path(__file__).resolve().parent
MODULE_DATA_DIR = MODULE_ROOT / "data"
MODULE_SLURM_SCRIPTS_DIR = MODULE_DATA_DIR / "slurm_scripts"
MODULE_SLURM_SCRIPTS_LIST = MODULE_DATA_DIR / "slurm_scripts.txt"
MODULE_PERF_DATA_XLSX = MODULE_DATA_DIR / "all_model_data.xlsx"

__all__ = [
    "SlurmHelperUtility",
    "JobScheduler",
    "REQUIRED_PERF_COLUMNS",
    "load_perf_data_for_jobs",
    "MODULE_ROOT",
    "MODULE_DATA_DIR",
    "MODULE_SLURM_SCRIPTS_DIR",
    "MODULE_SLURM_SCRIPTS_LIST",
    "MODULE_PERF_DATA_XLSX",
]
