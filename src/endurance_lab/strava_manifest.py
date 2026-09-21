from __future__ import annotations

import csv
import os
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from endurance_lab.config import project_path


DEFAULT_MANIFEST = Path("training_data/manifest.csv")
DEFAULT_DOWNLOAD_DIR = Path("training_data/import")
DEFAULT_COOKIES = Path("training_data/auth/cookies.txt")
DEFAULT_BROWSER_PROFILE = Path("training_data/auth/strava-browser-profile")

MANIFEST_FIELDS = [
    "strava_id",
    "activity_date",
    "sport",
    "name",
    "distance_m",
    "duration_seconds",
    "elevation_m",
    "source_device",
    "filename",
    "downloaded_format",
    "download_status",
    "downloaded_at",
    "sha256",
    "http_status",
    "result_status",
    "error",
    "attempts",
    "last_attempt_at",
]

ID_ALIASES = ("strava_id", "activity_id", "id")
ALIASES = {
    "activity_date": ("activity_date", "date", "started_at", "start_date", "start_time"),
    "sport": ("sport", "type", "activity_type", "sport_type"),
    "name": ("name", "activity_name", "title"),
    "distance_m": ("distance_m", "distance"),
    "duration_seconds": ("duration_seconds", "duration", "elapsed_time", "moving_time"),
    "elevation_m": ("elevation_m", "elevation", "total_elevation_gain"),
    "source_device": ("source_device", "device", "source"),
    "filename": ("filename", "downloaded_filename", "expected_filename"),
    "downloaded_format": ("downloaded_format", "format"),
    "download_status": ("download_status", "status"),
    "downloaded_at": ("downloaded_at",),
    "sha256": ("sha256", "file_hash"),
    "http_status": ("http_status",),
    "result_status": ("result_status", "result"),
    "error": ("error", "error_message"),
    "attempts": ("attempts", "download_attempts"),
    "last_attempt_at": ("last_attempt_at",),
}


@dataclass
class ManifestRow:
    strava_id: str
    activity_date: str = ""
    sport: str = ""
    name: str = ""
    distance_m: str = ""
    duration_seconds: str = ""
    elevation_m: str = ""
    source_device: str = ""
    filename: str = ""
    downloaded_format: str = ""
    download_status: str = "pending"
    downloaded_at: str = ""
    sha256: str = ""
    http_status: str = ""
    result_status: str = ""
    error: str = ""
    attempts: int = 0
    last_attempt_at: str = ""

    @classmethod
    def from_mapping(cls, values: dict[str, object]) -> "ManifestRow":
        normalized = {str(key).strip().lower(): _text(value) for key, value in values.items()}
        activity_id = _first(normalized, ID_ALIASES)
        validate_strava_id(activity_id)
        kwargs: dict[str, object] = {"strava_id": activity_id}
        for name, aliases in ALIASES.items():
            value = _first(normalized, aliases)
            kwargs[name] = _integer(value) if name == "attempts" else value
        if not kwargs["download_status"]:
            kwargs["download_status"] = "pending"
        return cls(**kwargs)

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)


def resolve_private_path(value: str | Path | None, default: Path) -> Path:
    return project_path(value or default)


def load_manifest(path: str | Path | None = None) -> list[ManifestRow]:
    target = resolve_private_path(path, DEFAULT_MANIFEST)
    if not target.exists():
        return []
    with target.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not any(name.strip().lower() in ID_ALIASES for name in reader.fieldnames):
            raise ValueError(f"Manifest has no Strava ID column: {target}")
        return [ManifestRow.from_mapping(dict(row)) for row in reader if any(row.values())]


def save_manifest(rows: list[ManifestRow], path: str | Path | None = None) -> Path:
    target = resolve_private_path(path, DEFAULT_MANIFEST)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_mapping())
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(target)
    return target


def merge_manifest_source(
    source: str | Path, manifest: str | Path | None = None
) -> tuple[Path, int, int]:
    target = resolve_private_path(manifest, DEFAULT_MANIFEST)
    existing = load_manifest(target)
    incoming = load_id_source(source)
    by_id = {row.strava_id: row for row in existing}
    added = updated = 0
    for candidate in incoming:
        current = by_id.get(candidate.strava_id)
        if current is None:
            existing.append(candidate)
            by_id[candidate.strava_id] = candidate
            added += 1
            continue
        changed = False
        for field in (
            "activity_date", "sport", "name", "distance_m", "duration_seconds",
            "elevation_m", "source_device",
        ):
            value = getattr(candidate, field)
            if value and value != getattr(current, field):
                setattr(current, field, value)
                changed = True
        if changed:
            updated += 1
    save_manifest(existing, target)
    return target, added, updated


def load_id_source(source: str | Path) -> list[ManifestRow]:
    path = project_path(source)
    if not path.is_file():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8-sig")
    first = next((line.strip() for line in text.splitlines() if line.strip()), "")
    header_names = {part.strip().lower() for part in first.split(",")}
    if "," in first or header_names.intersection(ID_ALIASES):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or not any(name.strip().lower() in ID_ALIASES for name in reader.fieldnames):
                raise ValueError(f"CSV has no Strava ID column: {path}")
            rows = [ManifestRow.from_mapping(dict(row)) for row in reader if any(row.values())]
    else:
        rows = []
        for line_number, raw in enumerate(text.splitlines(), start=1):
            value = raw.strip()
            if not value or value.startswith("#"):
                continue
            try:
                validate_strava_id(value)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            rows.append(ManifestRow(strava_id=value))
    unique: dict[str, ManifestRow] = {}
    for row in rows:
        unique.setdefault(row.strava_id, row)
    return list(unique.values())


def validate_strava_id(value: str) -> None:
    if not re.fullmatch(r"\d{7,20}", value or ""):
        raise ValueError(f"Invalid Strava activity ID: {value!r}")


def _first(values: dict[str, str], keys) -> str:
    return next((values[key] for key in keys if values.get(key)), "")


def _text(value: object) -> str:
    return "" if value is None else str(value).strip()


def _integer(value: str) -> int:
    try:
        return max(0, int(value or 0))
    except ValueError:
        return 0


assert [field.name for field in fields(ManifestRow)] == MANIFEST_FIELDS
