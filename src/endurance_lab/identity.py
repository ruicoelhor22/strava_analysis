from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path


def source_activity_id_from_filename(filename: str | Path) -> str | None:
    candidates = re.findall(r"(?<!\d)(\d{7,12})(?!\d)", Path(filename).name)
    return max(candidates, key=len) if candidates else None


def deterministic_identity(started_at: datetime, sport: str) -> str:
    rounded = started_at.replace(microsecond=0).isoformat()
    return f"activity:{rounded}:{sport}"
