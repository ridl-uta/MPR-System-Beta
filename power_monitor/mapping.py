from __future__ import annotations

import json
from pathlib import Path

from .models import MappingConfig, PDUConfig


def default_mapping_path() -> Path:
    """Return module-local default mapping path."""
    return Path(__file__).resolve().parent / "data" / "mapping.json"


def load_mapping(map_path: str | Path | None = None) -> MappingConfig:
    path = Path(map_path) if map_path else default_mapping_path()
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    outlets = int(raw.get("outlets", 24))
    pdus: list[PDUConfig] = []
    for item in raw.get("pdus", []):
        host = item.get("host")
        if not host:
            continue
        port = int(item.get("port", 23))
        outlet_map = {int(k): str(v) for k, v in item.get("map", {}).items()}
        pdus.append(PDUConfig(host=host, port=port, outlet_map=outlet_map))

    return MappingConfig(outlets=outlets, pdus=pdus)


def build_poll_command(outlets: int) -> str:
    return f"olReading 1-{int(outlets)} power"
