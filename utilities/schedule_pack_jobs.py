#!/usr/bin/env python3
"""Submit packs of Slurm scripts.

Per-node splits sum to CPUS_PER_NODE.
"""

import argparse
import csv
import os
import random
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple


# ==== Default Slurm scripts file ====
DEFAULT_SCRIPTS_FILE = "slurm_scripts.txt"

# ==== Defaults ====
DEFAULT_PARTITION = "debug"
DEFAULT_TIME_LIMIT = "00:20:00"
DEFAULT_PACKS = 1
DEFAULT_MAX_JOBS_PER_PACK = 4
DEFAULT_MAX_NODES_PER_JOB = 2
DEFAULT_LOG_DIR = "/shared/job_logs"


class JobScheduler:
    def __init__(self):
        self.nodes = None
        self.cpus_per_node = None
        self.partition = os.getenv("PARTITION", DEFAULT_PARTITION)
        self.time_limit = os.getenv("TIME_LIMIT", DEFAULT_TIME_LIMIT)
        self.account = os.getenv("ACCOUNT", "")
        self.packs = DEFAULT_PACKS
        self.max_jobs_per_pack = DEFAULT_MAX_JOBS_PER_PACK
        self.max_nodes_per_job = DEFAULT_MAX_NODES_PER_JOB
        self.dry_run = False
        self.log_dir = os.getenv("LOG_DIR", DEFAULT_LOG_DIR)
        self.csv_out = ""
        self.task_splits = os.getenv("TASK_SPLITS", "")
        self.scripts_file = DEFAULT_SCRIPTS_FILE
        self.slurm_scripts = []

    def parse_args(self):
        epilog = (
            "Environment variables:\n"
            '  TASK_SPLITS="a,b,c"  Force per-node splits; sum must match '
            "CPUS_PER_NODE\n"
            "  LOG_DIR=/path         Where to write logs/CSV "
            f"(defaults to {DEFAULT_LOG_DIR}; falls back to $HOME/job_logs)\n"
            "  PARTITION             Default partition\n"
            "  TIME_LIMIT            Default time limit\n"
            "  ACCOUNT               Default Slurm account\n"
        )
        parser = argparse.ArgumentParser(
            description="Submit packs of Slurm scripts with per-node splits",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=epilog,
        )

        # Required arguments
        parser.add_argument(
            "-N",
            "--nodes",
            type=int,
            required=True,
            help="Nodes available per job (upper bound)",
        )
        parser.add_argument(
            "-C",
            "--cpus-per-node",
            type=int,
            required=True,
            help="Cores per node (e.g., 20)",
        )

        # Optional arguments
        parser.add_argument(
            "-P",
            "--partition",
            default=self.partition,
            help=f"Partition (default: {self.partition})",
        )
        parser.add_argument(
            "-T",
            "--time",
            dest="time_limit",
            default=self.time_limit,
            help=f"Time limit (default: {self.time_limit})",
        )
        parser.add_argument(
            "-A",
            "--account",
            default=self.account,
            help="Slurm account",
        )
        parser.add_argument(
            "-p",
            "--packs",
            type=int,
            default=self.packs,
            help=f"Number of packs (default: {self.packs})",
        )
        parser.add_argument(
            "-m",
            "--max-per-pack",
            type=int,
            default=self.max_jobs_per_pack,
            help=f"Max jobs per pack (default: {self.max_jobs_per_pack})",
        )
        parser.add_argument(
            "--max-nodes-per-job",
            type=int,
            default=self.max_nodes_per_job,
            help=f"Limit nodes per job (default: {self.max_nodes_per_job})",
        )
        parser.add_argument(
            "--scripts-file",
            default=self.scripts_file,
            help="File containing list of Slurm scripts",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Print submissions, don't submit",
        )

        args = parser.parse_args()

        # Validation
        if args.nodes < 1:
            parser.error("Invalid -N: must be >= 1")
        if args.cpus_per_node < 1:
            parser.error("Invalid -C: must be >= 1")
        if args.packs < 1:
            parser.error("Invalid packs: must be >= 1")
        if args.max_per_pack < 1:
            parser.error("Invalid max-per-pack: must be >= 1")
        if args.max_nodes_per_job < 1:
            parser.error("Invalid --max-nodes-per-job: must be >= 1")

        # Set instance variables
        self.nodes = args.nodes
        self.cpus_per_node = args.cpus_per_node
        self.partition = args.partition
        self.time_limit = args.time_limit
        self.account = args.account
        self.packs = args.packs
        self.max_jobs_per_pack = args.max_per_pack
        self.max_nodes_per_job = args.max_nodes_per_job
        self.dry_run = args.dry_run
        self.scripts_file = args.scripts_file

    def require_cmd(self, cmd: str):
        """Check if command exists in PATH"""
        if not shutil.which(cmd):
            print(f"[ERROR] '{cmd}' not found in PATH")
            sys.exit(127)

    def load_slurm_scripts(self):
        """Load Slurm scripts from file"""
        scripts_path = Path(self.scripts_file)

        if not scripts_path.exists():
            print(f"[ERROR] Scripts file not found: {self.scripts_file}")
            print(
                f"[INFO] Create {self.scripts_file} with script paths, "
                "one per line."
            )
            print("       e.g.:")
            print("         /shared/run_minife.slurm")
            print("         /shared/run_comd.slurm")
            sys.exit(4)

        try:
            with open(scripts_path, 'r') as f:
                lines = f.readlines()
        except IOError as e:
            print(f"[ERROR] Cannot read scripts file {self.scripts_file}: {e}")
            sys.exit(4)

        # Parse lines, skip comments and empty lines
        scripts = []
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Handle quoted paths
            if (
                (line.startswith('"') and line.endswith('"'))
                or (line.startswith("'") and line.endswith("'"))
            ):
                line = line[1:-1]

            scripts.append(line)

        if not scripts:
            print(f"[ERROR] No scripts found in {self.scripts_file}")
            print("[INFO] File should contain one script path per line.")
            print("       Lines starting with # are ignored.")
            sys.exit(4)

        self.slurm_scripts = scripts
        print(f"[INFO] Loaded {len(scripts)} scripts from {self.scripts_file}")

    def validate_scripts(self):
        """Ensure all Slurm scripts exist"""
        if not self.slurm_scripts:
            print("[ERROR] No Slurm scripts loaded")
            sys.exit(4)

        missing_scripts = []
        for script in self.slurm_scripts:
            if not Path(script).is_file():
                missing_scripts.append(script)

        if missing_scripts:
            print("[ERROR] Missing script(s):")
            for script in missing_scripts:
                print(f"  {script}")
            sys.exit(4)

    def prepare_logs(self):
        """Prepare log directory with fallback"""
        target = self.log_dir

        try:
            Path(target).mkdir(parents=True, exist_ok=True)
            # Test if writable
            test_file = Path(target) / ".write_test"
            test_file.touch()
            test_file.unlink()
        except (OSError, PermissionError):
            target = str(Path.home() / "job_logs")
            Path(target).mkdir(parents=True, exist_ok=True)
            print(f"[WARN] LOG_DIR not writable; falling back to {target}")

        self.log_dir = target
        self.csv_out = str(Path(target) / "jobs_submitted.csv")

        # Create CSV with header if it doesn't exist
        if not Path(self.csv_out).exists():
            with open(self.csv_out, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "pack_id", "job_idx", "jobid",
                    "nodes", "tasks_per_node", "ntasks", "script_path"
                ])

    def rand_between(self, a: int, b: int) -> int:
        """Random integer between a and b inclusive"""
        return random.randint(a, b)

    def random_composition_fixed_k(self, total: int, k: int) -> List[int]:
        """Generate random composition of total into k parts"""
        if k < 1 or total < k:
            print("[ERROR] bad composition k/total")
            sys.exit(3)

        parts = []
        remaining = total

        for i in range(k - 1):
            max_for_this = remaining - (k - i - 1)
            val = self.rand_between(1, max_for_this)
            parts.append(val)
            remaining -= val

        parts.append(remaining)
        return parts

    def random_composition(self, total: int, max_parts: int) -> List[int]:
        """Generate random composition of total with at most max_parts parts"""
        if total == 1:
            return [1]

        kmax = min(max_parts, total)
        k = self.rand_between(2, kmax)
        return self.random_composition_fixed_k(total, k)

    def pick_random_script(self) -> str:
        """Pick a random Slurm script"""
        return random.choice(self.slurm_scripts)

    def sanitize_job_component(self, s: str) -> str:
        """Sanitize string for use in job names"""
        s = Path(s).stem  # Remove path and extension
        s = re.sub(r'[^A-Za-z0-9._-]', '_', s)  # Replace invalid chars with _
        return s

    def csv_append(self, row: str):
        """Append a row to the CSV file"""
        with open(self.csv_out, 'a') as f:
            f.write(row + '\n')

    def submit_script_job(
        self,
        job_name: str,
        script_path: str,
        nodes: int,
        tpn: int,
    ) -> Tuple[bool, str]:
        """Submit a job via sbatch"""
        ntasks = nodes * tpn

        sbatch_cmd = [
            "sbatch", "--parsable",
            "-J", job_name,
            "-N", str(nodes),
            "-n", str(ntasks),
            "--ntasks-per-node", str(tpn),
            "--cpus-per-task", "1",
            "-p", self.partition,
            "-t", self.time_limit,
            "--output", f"{self.log_dir}/slurm-%x-%j.out"
        ]

        if self.account:
            sbatch_cmd.extend(["-A", self.account])

        sbatch_cmd.append(script_path)

        if self.dry_run:
            print(f"[DRY-RUN] {' '.join(sbatch_cmd)}")
            return True, "DRYRUN"

        try:
            result = subprocess.run(
                sbatch_cmd,
                capture_output=True,
                text=True,
                check=True,
            )
            job_id = result.stdout.strip()
            return True, job_id
        except subprocess.CalledProcessError:
            print(f"[ERROR] sbatch failed for {script_path}")
            return False, ""

    def parse_task_splits(self) -> List[int]:
        """Parse TASK_SPLITS environment variable"""
        if not self.task_splits:
            return []

        try:
            splits = [int(x.strip()) for x in self.task_splits.split(',')]
        except ValueError as e:
            print(f"[ERROR] bad TASK_SPLITS format: {e}")
            sys.exit(3)

        for s in splits:
            if s < 1:
                print(f"[ERROR] bad TASK_SPLITS element: {s}")
                sys.exit(3)

        total = sum(splits)
        if total != self.cpus_per_node:
            print(
                f"[ERROR] TASK_SPLITS must sum to {self.cpus_per_node} "
                f"(got {total})"
            )
            sys.exit(3)

        return splits

    def run(self):
        """Main execution function"""
        self.parse_args()
        self.require_cmd("sbatch")
        self.load_slurm_scripts()
        self.validate_scripts()
        self.prepare_logs()

        print(
            f"[INFO] Nodes={self.nodes}  CPUs/Node={self.cpus_per_node}  "
            f"Packs={self.packs}"
        )
        print(f"[INFO] Partition={self.partition}  Time={self.time_limit}")
        if self.account:
            print(f"[INFO] Account={self.account}")
        print(f"[INFO] Max nodes per job: {self.max_nodes_per_job}")
        if self.task_splits:
            print(f"[INFO] Using fixed per-node splits: {self.task_splits}")

        for pack in range(1, self.packs + 1):
            print(f"=== Pack {pack} ===")

            # Determine per-node task splits
            if self.task_splits:
                pernode_tasks = self.parse_task_splits()
            else:
                pernode_tasks = self.random_composition(
                    self.cpus_per_node,
                    self.max_jobs_per_pack,
                )

            # Show composition
            pretty = " + ".join(str(t) for t in pernode_tasks)
            print(
                f"[INFO] Pack {pack} split per node: {pretty} = "
                f"{self.cpus_per_node}"
            )

            for idx, tpn in enumerate(pernode_tasks, 1):
                script_path = self.pick_random_script()
                short = self.sanitize_job_component(script_path)

                # Clamp per-job nodes to [1, min(MAX_NODES_PER_JOB, NODES)]
                maxn = min(self.max_nodes_per_job, self.nodes)
                job_nodes = self.rand_between(1, maxn)

                job_name = f"pack{pack}-{short}-tpn{tpn}n{job_nodes}"

                success, job_id = self.submit_script_job(
                    job_name,
                    script_path,
                    job_nodes,
                    tpn,
                )
                if not success:
                    continue

                # Log to CSV
                timestamp = datetime.now().isoformat()
                ntasks = job_nodes * tpn
                row = ",".join(
                    [
                        timestamp,
                        str(pack),
                        str(idx),
                        job_id,
                        str(job_nodes),
                        str(tpn),
                        str(ntasks),
                        script_path,
                    ]
                )
                self.csv_append(row)

                print(
                    f"[OK] {job_name} -> {job_id} (nodes={job_nodes}, "
                    f"tpn={tpn}, ntasks={ntasks}, script={script_path})"
                )

        print(f"[DONE] Logged submissions to: {self.csv_out}")


def main():
    scheduler = JobScheduler()
    scheduler.run()


if __name__ == "__main__":
    main()
