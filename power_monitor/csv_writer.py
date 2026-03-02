from __future__ import annotations

import csv
from pathlib import Path


def append_power_csv(csv_path: str, ts_iso: str, node_totals: dict[str, int]) -> None:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    need_header = (not path.exists()) or (path.stat().st_size == 0)
    header_nodes: list[str] = []

    if not need_header:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, [])
            if header and header[0].lower() == "timestamp":
                header_nodes = header[1:-1]
            else:
                need_header = True

    if need_header:
        header_nodes = sorted(node_totals.keys())
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", *header_nodes, "total_watts"])

    total = sum(node_totals.get(node, 0) for node in header_nodes)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([ts_iso, *[node_totals.get(node, 0) for node in header_nodes], total])
