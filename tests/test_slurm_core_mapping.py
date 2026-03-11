from __future__ import annotations

import subprocess
import unittest

from dvfs.controller import DVFSController
from dvfs.job_utilities import (
    expand_list_to_ints,
    map_cpu_ids_to_core_ids as dvfs_map_cpu_ids_to_core_ids,
    parse_cpu_core_map as dvfs_parse_cpu_core_map,
)
from job_scheduler.helper_utility import SlurmHelperUtility


class _FakeSlurmHelper(SlurmHelperUtility):
    def __init__(self, probe_stdout: str) -> None:
        super().__init__()
        self._probe_stdout = probe_stdout

    def get_job_nodelist_and_state(self, job_id: str) -> tuple[str, str]:
        _ = job_id
        return "ridlserver04,ridlserver12", "RUNNING"

    def expand_nodelist(self, compact_nodelist: str) -> list[str]:
        _ = compact_nodelist
        return ["ridlserver04", "ridlserver12"]

    def run_command(self, cmd: list[str]) -> subprocess.CompletedProcess[str]:
        _ = cmd
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout=self._probe_stdout,
            stderr="",
        )


class TestCpuToCoreMapping(unittest.TestCase):
    def test_helper_mapping_deduplicates_smt_siblings(self) -> None:
        cpu_ids = [0, 1, 2, 3, 20, 21, 22, 23]
        cpu_to_core = SlurmHelperUtility.parse_cpu_core_map(
            "\n".join(
                [
                    "0,0",
                    "1,1",
                    "2,2",
                    "3,3",
                    "20,0",
                    "21,1",
                    "22,2",
                    "23,3",
                ]
            )
        )
        self.assertEqual(
            SlurmHelperUtility.map_cpu_ids_to_core_ids(cpu_ids, cpu_to_core),
            [0, 1, 2, 3],
        )

    def test_dvfs_job_utilities_mapping_matches_helper(self) -> None:
        cpu_ids = expand_list_to_ints("0-3,20-23")
        cpu_to_core = dvfs_parse_cpu_core_map(
            "\n".join(
                [
                    "0,0",
                    "1,1",
                    "2,2",
                    "3,3",
                    "20,0",
                    "21,1",
                    "22,2",
                    "23,3",
                ]
            )
        )
        self.assertEqual(
            dvfs_map_cpu_ids_to_core_ids(cpu_ids, cpu_to_core),
            [0, 1, 2, 3],
        )

    def test_collect_job_cores_translates_cpu_ids_to_core_ids(self) -> None:
        probe_stdout = "\n".join(
            [
                "ridlserver04\t0-3,20-23\t0:0,1:1,2:2,3:3,20:0,21:1,22:2,23:3",
                "ridlserver12\t4-5,24-25\t4:4,5:5,24:4,25:5",
            ]
        )
        helper = _FakeSlurmHelper(probe_stdout)
        state, cores_by_node = helper.collect_job_cores("9999")

        self.assertEqual(state, "RUNNING")
        self.assertEqual(
            cores_by_node,
            {
                "ridlserver04": [0, 1, 2, 3],
                "ridlserver12": [4, 5],
            },
        )


class TestDvfsVerification(unittest.TestCase):
    def test_readback_uses_only_matching_target_ids(self) -> None:
        stdout = "\n".join(
            [
                "GEOPMREAD CPU_FREQUENCY_MAX_CONTROL core 0 2.1e+09",
                "GEOPMREAD CPU_FREQUENCY_MAX_CONTROL core 1 2.1e+09",
                "GEOPMREAD CPU_FREQUENCY_MAX_CONTROL core 2 3.2e+09",
            ]
        )
        values, source, signal, reason = DVFSController._extract_readback_values(
            stdout,
            target_indices=[0, 1],
        )
        self.assertEqual(values, [2100.0, 2100.0])
        self.assertEqual(source, "geopmread")
        self.assertEqual(signal, "CPU_FREQUENCY_MAX_CONTROL")
        self.assertEqual(reason, "ok")

    def test_readback_target_mismatch_does_not_fallback_to_all_values(self) -> None:
        stdout = "\n".join(
            [
                "GEOPMREAD CPU_FREQUENCY_MAX_CONTROL core 10 2.1e+09",
                "GEOPMREAD CPU_FREQUENCY_MAX_CONTROL core 11 3.2e+09",
            ]
        )
        values, source, signal, reason = DVFSController._extract_readback_values(
            stdout,
            target_indices=[0, 1],
        )
        self.assertEqual(values, [])
        self.assertEqual(source, "geopmread")
        self.assertEqual(signal, "CPU_FREQUENCY_MAX_CONTROL")
        self.assertEqual(reason, "target_ids_not_found_in_readback")

    def test_verification_fields_fail_on_target_id_mismatch(self) -> None:
        controller = DVFSController()
        stdout = "\n".join(
            [
                "GEOPMREAD CPU_FREQUENCY_MAX_CONTROL core 10 2.1e+09",
                "GEOPMREAD CPU_FREQUENCY_MAX_CONTROL core 11 2.1e+09",
            ]
        )
        fields = controller._build_verification_fields(
            requested_mhz=2100.0,
            status="APPLIED",
            returncode=0,
            stdout=stdout,
            target_core_numbers=[0, 1],
        )
        self.assertEqual(fields["verify_status"], "FAIL")
        self.assertEqual(fields["verify_reason"], "target_ids_not_found_in_readback")


if __name__ == "__main__":
    unittest.main(verbosity=2)
