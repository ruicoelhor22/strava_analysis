from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from endurance_lab.config import project_path


class SyncAlreadyRunning(RuntimeError):
    pass


class SyncLock:
    def __init__(self, path: str | Path = "training_data/sync.lock", stale_hours: int = 6) -> None:
        self.path = project_path(path)
        self.stale_after = timedelta(hours=stale_hours)
        self.acquired = False

    def __enter__(self) -> "SyncLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._clear_stale()
        payload = json.dumps({
            "pid": os.getpid(),
            "started_at": datetime.now(timezone.utc).isoformat(),
        }).encode("utf-8")
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise SyncAlreadyRunning("Sync already running.") from exc
        try:
            os.write(descriptor, payload)
        finally:
            os.close(descriptor)
        self.acquired = True
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self.acquired:
            try:
                self.path.unlink(missing_ok=True)
            finally:
                self.acquired = False

    def _clear_stale(self) -> None:
        if not self.path.exists():
            return
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            started = datetime.fromisoformat(str(value.get("started_at")))
            pid = int(value.get("pid") or 0)
        except Exception:
            started, pid = datetime.fromtimestamp(0, timezone.utc), 0
        age = datetime.now(timezone.utc) - started.astimezone(timezone.utc)
        if age <= self.stale_after and _pid_exists(pid):
            return
        self.path.unlink(missing_ok=True)


def _pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False
