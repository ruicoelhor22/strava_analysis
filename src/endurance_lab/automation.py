from __future__ import annotations

import json
import os
import secrets
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import requests

from endurance_lab.config import PROJECT_ROOT, project_path


SYNC_TASK = "Endurance Lab Sync"
DASHBOARD_TASK = "Endurance Lab Dashboard"
RUNTIME_FILE = "training_data/automation/runtime.json"
NGROK_EXE = "training_data/tools/ngrok/ngrok.exe"
NGROK_CONFIG = "training_data/auth/ngrok.yml"
NGROK_POLICY = "training_data/auth/ngrok-policy.yml"


@dataclass
class TaskInstallResult:
    sync_task: str
    dashboard_task: str
    sync_minutes: int
    runtime_file: str
    commands: list[list[str]]
    installed: bool
    dashboard_method: str


def install_windows_tasks(sync_minutes: int = 60, *, dry_run: bool = False) -> TaskInstallResult:
    if os.name != "nt":
        raise RuntimeError("Windows Task Scheduler installation is only available on Windows")
    sync_minutes = max(15, min(int(sync_minutes), 1439))
    runtime = project_path(RUNTIME_FILE)
    sync_script = PROJECT_ROOT / "scripts" / "run_sync.ps1"
    dashboard_script = PROJECT_ROOT / "scripts" / "run_dashboard.ps1"
    if not sync_script.exists() or not dashboard_script.exists():
        raise RuntimeError("Automation launcher scripts are missing")
    start = (datetime.now() + timedelta(minutes=2)).strftime("%H:%M")
    commands = [
        [
            "schtasks.exe", "/Create", "/F", "/TN", SYNC_TASK,
            "/SC", "MINUTE", "/MO", str(sync_minutes), "/ST", start,
            "/TR", _powershell_action(sync_script), "/RL", "LIMITED",
        ],
        [
            "schtasks.exe", "/Create", "/F", "/TN", DASHBOARD_TASK,
            "/SC", "ONLOGON", "/TR", _powershell_action(dashboard_script), "/RL", "LIMITED",
        ],
    ]
    if not dry_run:
        runtime.parent.mkdir(parents=True, exist_ok=True)
        value = _runtime_value(runtime)
        value.update({
            "project_root": str(PROJECT_ROOT),
            "python_executable": sys.executable,
            "installed_at": datetime.now().astimezone().isoformat(),
            "sync_minutes": sync_minutes,
        })
        runtime.write_text(json.dumps(value, indent=2), encoding="utf-8")
        for index, command in enumerate(commands):
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
            if completed.returncode:
                if index == 1:
                    _write_dashboard_startup(dashboard_script)
                    continue
                raise RuntimeError(
                    f"Task Scheduler rejected {command[4]!r}: "
                    f"{(completed.stderr or completed.stdout).strip()}"
                )
        value["sync_task_installed"] = True
        value["dashboard_startup_installed"] = True
        runtime.write_text(json.dumps(value, indent=2), encoding="utf-8")
    return TaskInstallResult(
        SYNC_TASK, DASHBOARD_TASK, sync_minutes, str(runtime), commands, not dry_run,
        "scheduled_task" if dry_run or _task_exists(DASHBOARD_TASK) else "startup_folder",
    )


def uninstall_windows_tasks(*, dry_run: bool = False) -> dict[str, Any]:
    commands = [
        ["schtasks.exe", "/Delete", "/F", "/TN", SYNC_TASK],
        ["schtasks.exe", "/Delete", "/F", "/TN", DASHBOARD_TASK],
    ]
    if not dry_run:
        for command in commands:
            subprocess.run(command, capture_output=True, text=True, check=False)
        startup = _dashboard_startup_path()
        if startup.exists():
            startup.unlink()
        tunnel_startup = _tunnel_startup_path()
        if tunnel_startup.exists():
            tunnel_startup.unlink()
        runtime = project_path(RUNTIME_FILE)
        value = _runtime_value(runtime)
        value["sync_task_installed"] = False
        value["dashboard_startup_installed"] = False
        if runtime.exists():
            runtime.write_text(json.dumps(value, indent=2), encoding="utf-8")
    return {"removed": not dry_run, "tasks": [SYNC_TASK, DASHBOARD_TASK], "commands": commands}


def automation_status() -> dict[str, Any]:
    runtime = project_path(RUNTIME_FILE)
    runtime_value = _runtime_value(runtime)
    sync_installed = _task_exists(SYNC_TASK) or bool(runtime_value.get("sync_task_installed"))
    dashboard_installed = (
        _task_exists(DASHBOARD_TASK) or _dashboard_startup_path().exists()
        or bool(runtime_value.get("dashboard_startup_installed"))
    )
    return {
        "platform": os.name,
        "runtime_configured": runtime.exists(),
        "runtime_file": str(runtime),
        "sync_task": sync_installed,
        "dashboard_task": dashboard_installed,
        "dashboard_startup_method": (
            "scheduled_task" if _task_exists(DASHBOARD_TASK) else
            "startup_folder" if _dashboard_startup_path().exists() else None
        ),
        "phone_access": phone_access_status(),
    }


def phone_access_status(port: int = 8501) -> dict[str, Any]:
    executable = _tailscale_executable()
    if not executable:
        ngrok = _ngrok_status()
        if ngrok:
            return ngrok
        configured = project_path(NGROK_CONFIG).exists() and project_path(NGROK_POLICY).exists()
        return {
            "method": "ngrok", "tailscale_installed": False,
            "ngrok_installed": project_path(NGROK_EXE).exists(), "configured": configured,
            "connected": False, "ip": None, "dashboard_url": None,
            "message": "Run phone-access-setup with an ngrok account token." if not configured else
                       "ngrok is configured but not running; sign out and in, or restart the launcher.",
        }
    completed = subprocess.run(
        [str(executable), "ip", "-4"], capture_output=True, text=True, check=False, timeout=10
    )
    address = completed.stdout.strip().splitlines()[0] if completed.returncode == 0 and completed.stdout.strip() else None
    return {
        "method": "tailscale", "tailscale_installed": True, "ngrok_installed": False,
        "configured": True, "connected": bool(address), "ip": address,
        "dashboard_url": f"http://{address}:{port}" if address else None,
        "message": "Open the dashboard URL from a phone connected to the same tailnet." if address else
                   "Tailscale is installed but not connected; sign in to Tailscale.",
    }


def configure_ngrok_phone_access(authtoken: str, username: str = "endurance") -> dict[str, Any]:
    token = authtoken.strip()
    if not token:
        raise ValueError("An ngrok authtoken is required")
    if token.startswith(("cr_", "ak_")):
        raise ValueError(
            "This is an ngrok API credential, not an agent authtoken. "
            "Copy the value from https://dashboard.ngrok.com/get-started/your-authtoken"
        )
    if not username.replace("-", "").replace("_", "").isalnum():
        raise ValueError("Username may contain letters, numbers, hyphens, and underscores")
    executable = project_path(NGROK_EXE)
    if not executable.exists():
        raise FileNotFoundError(f"Portable ngrok executable not found: {executable}")
    config = project_path(NGROK_CONFIG)
    policy = project_path(NGROK_POLICY)
    config.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [str(executable), "config", "add-authtoken", token, "--config", str(config)],
        capture_output=True, text=True, check=False,
    )
    if completed.returncode:
        raise RuntimeError((completed.stderr or completed.stdout).strip())
    password = secrets.token_urlsafe(24)
    policy.write_text(
        "on_http_request:\n"
        "  - actions:\n"
        "      - type: basic-auth\n"
        "        config:\n"
        "          realm: Endurance Lab\n"
        "          credentials:\n"
        f"            - {username}:{password}\n"
        "          enforce: true\n",
        encoding="utf-8",
    )
    runtime = project_path(RUNTIME_FILE)
    runtime.parent.mkdir(parents=True, exist_ok=True)
    value = _runtime_value(runtime)
    value.update({
        "project_root": str(PROJECT_ROOT), "python_executable": sys.executable,
        "ngrok_executable": str(executable), "ngrok_config": str(config),
        "ngrok_policy": str(policy), "phone_username": username,
    })
    runtime.write_text(json.dumps(value, indent=2), encoding="utf-8")
    _write_tunnel_startup(PROJECT_ROOT / "scripts" / "run_tunnel.ps1")
    subprocess.Popen(
        ["powershell.exe", "-NoProfile", "-WindowStyle", "Hidden", "-ExecutionPolicy", "Bypass",
         "-File", str(PROJECT_ROOT / "scripts" / "run_tunnel.ps1")],
        cwd=PROJECT_ROOT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    return {
        "configured": True, "username": username, "password": password,
        "message": "Save this password now; it is shown only once. Check phone-access after the tunnel connects.",
    }


def _task_exists(name: str) -> bool:
    if os.name != "nt":
        return False
    completed = subprocess.run(
        ["schtasks.exe", "/Query", "/TN", name], capture_output=True, text=True, check=False
    )
    return completed.returncode == 0


def _powershell_action(script: Path) -> str:
    return f'powershell.exe -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File "{script}"'


def _dashboard_startup_path() -> Path:
    appdata = Path(os.environ.get("APPDATA", ""))
    return appdata / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup" / "Endurance Lab Dashboard.cmd"


def _tunnel_startup_path() -> Path:
    return _dashboard_startup_path().with_name("Endurance Lab Phone Access.cmd")


def _write_dashboard_startup(script: Path) -> None:
    target = _dashboard_startup_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        '@echo off\r\n'
        f'start "" /min powershell.exe -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File "{script}"\r\n',
        encoding="utf-8",
    )


def _write_tunnel_startup(script: Path) -> None:
    target = _tunnel_startup_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        '@echo off\r\n'
        f'start "" /min powershell.exe -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File "{script}"\r\n',
        encoding="utf-8",
    )


def _runtime_value(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _ngrok_status() -> dict[str, Any] | None:
    try:
        response = requests.get("http://127.0.0.1:4040/api/tunnels", timeout=1.0)
        response.raise_for_status()
        tunnels = response.json().get("tunnels", [])
        public = next((item.get("public_url") for item in tunnels if item.get("proto") == "https"), None)
    except (requests.RequestException, ValueError, AttributeError):
        return None
    if not public:
        return None
    return {
        "method": "ngrok", "tailscale_installed": False, "ngrok_installed": True,
        "configured": True, "connected": True, "ip": None, "dashboard_url": public,
        "message": "Open this HTTPS URL on your phone and enter the private Endurance Lab credentials.",
    }


def _tailscale_executable() -> Path | None:
    discovered = shutil.which("tailscale") or shutil.which("tailscale.exe")
    candidates = [
        Path(discovered) if discovered else None,
        Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Tailscale" / "tailscale.exe",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Tailscale" / "tailscale.exe",
    ]
    return next((path for path in candidates if path and path.exists()), None)
