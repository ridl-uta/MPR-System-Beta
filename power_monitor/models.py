from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PDUConfig:
    host: str
    port: int
    outlet_map: dict[int, str]


@dataclass(frozen=True)
class MappingConfig:
    outlets: int
    pdus: list[PDUConfig]
