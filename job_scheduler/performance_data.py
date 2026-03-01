from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_PERF_COLUMNS = ["Resource Reduction", "Extra Execution", "Power"]


def load_perf_data_for_jobs(
    *,
    xlsx_path: Path,
    job_names: list[str],
    sheet_map: dict[str, str],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Load workbook data and keep only columns required by MPR."""
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Workbook not found: {xlsx_path}")

    xl = pd.ExcelFile(xlsx_path)
    available = set(xl.sheet_names)

    jobs: dict[str, pd.DataFrame] = {}
    audit_rows: list[dict[str, object]] = []

    for job_name in job_names:
        sheet = sheet_map.get(job_name, job_name)
        if sheet not in available:
            audit_rows.append({"job": job_name, "sheet": sheet, "status": "MISSING_SHEET"})
            continue

        raw = pd.read_excel(xlsx_path, sheet_name=sheet)
        missing = [column for column in REQUIRED_PERF_COLUMNS if column not in raw.columns]
        if missing:
            audit_rows.append(
                {
                    "job": job_name,
                    "sheet": sheet,
                    "status": "MISSING_COLUMNS",
                    "details": ",".join(missing),
                }
            )
            continue

        model_df = raw[REQUIRED_PERF_COLUMNS].dropna().reset_index(drop=True)
        if model_df.empty:
            audit_rows.append({"job": job_name, "sheet": sheet, "status": "EMPTY_AFTER_CLEAN"})
            continue

        jobs[job_name] = model_df
        audit_rows.append(
            {"job": job_name, "sheet": sheet, "status": "LOADED", "rows": len(model_df)}
        )

    return jobs, pd.DataFrame(audit_rows)
