from __future__ import annotations

import re

OUTLET_LINE_RE = re.compile(r"^\s*(\d+):.*?(\d+)\s*W\s*$")


def parse_outlet_readings(text: str) -> dict[int, int]:
    values: dict[int, int] = {}
    for line in text.splitlines():
        match = OUTLET_LINE_RE.match(line)
        if match:
            values[int(match.group(1))] = int(match.group(2))
    return values
