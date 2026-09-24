$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$runtimePath = Join-Path $projectRoot "training_data\automation\runtime.json"
$logDirectory = Join-Path $projectRoot "training_data\logs"
$logPath = Join-Path $logDirectory "dashboard.log"

if (-not (Test-Path -LiteralPath $runtimePath)) {
    throw "Automation runtime file is missing. Run: python -m endurance_lab automation-install"
}

$runtime = Get-Content -Raw -LiteralPath $runtimePath | ConvertFrom-Json
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
if ((Test-Path -LiteralPath $logPath) -and (Get-Item -LiteralPath $logPath).Length -gt 5MB) {
    Move-Item -Force -LiteralPath $logPath -Destination "$logPath.1"
}
Set-Location -LiteralPath $runtime.project_root

"[$(Get-Date -Format o)] Dashboard starting" | Add-Content -LiteralPath $logPath
& $runtime.python_executable -m streamlit run dashboard/app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true 2>&1 | Add-Content -LiteralPath $logPath
