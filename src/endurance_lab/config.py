from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    database: Path
    import_dir: Path
    raw_dir: Path
    athlete_config: Path


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def paths() -> Paths:
    private = project_path(os.getenv("ENDURANCE_ATHLETE_CONFIG", "config/athlete.yaml"))
    example = PROJECT_ROOT / "config" / "athlete.example.yaml"
    return Paths(
        database=project_path(os.getenv("ENDURANCE_DB_PATH", "data/endurance_lab.sqlite3")),
        import_dir=project_path(os.getenv("ENDURANCE_IMPORT_DIR", "data/import")),
        raw_dir=project_path(os.getenv("ENDURANCE_RAW_DIR", "data/raw")),
        athlete_config=private if private.exists() else example,
    )


def load_athlete_config(path: str | Path | None = None) -> dict[str, Any]:
    resolved = project_path(path) if path else paths().athlete_config
    with resolved.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Athlete config must be a mapping: {resolved}")
    return deepcopy(loaded)


def ftp_at(config: dict[str, Any], when: datetime | str | None) -> float | None:
    history = config.get("cycling", {}).get("ftp_history", [])
    if not history:
        return None
    target = _date_text(when) or "9999-12-31"
    eligible = [item for item in history if str(item.get("effective_from", "")) <= target]
    if not eligible:
        return None
    selected = max(eligible, key=lambda item: str(item.get("effective_from", "")))
    value = selected.get("watts")
    return float(value) if value is not None else None


def _date_text(value: datetime | str | None) -> str | None:
    if value is None:
        return None
    return value.date().isoformat() if isinstance(value, datetime) else str(value)[:10]


def ensure_local_layout() -> Paths:
    resolved = paths()
    resolved.database.parent.mkdir(parents=True, exist_ok=True)
    resolved.import_dir.mkdir(parents=True, exist_ok=True)
    resolved.raw_dir.mkdir(parents=True, exist_ok=True)
    private_config = PROJECT_ROOT / "config" / "athlete.yaml"
    if not private_config.exists():
        private_config.write_text(resolved.athlete_config.read_text(encoding="utf-8"), encoding="utf-8")
    return paths()
