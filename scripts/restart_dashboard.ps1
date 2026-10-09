$ErrorActionPreference = 'Stop'
$dashboardRoot = Split-Path -Parent $PSScriptRoot
$dashboardRuntime = Get-Content -LiteralPath (Join-Path $dashboardRoot 'training_data/automation/runtime.json') -Raw | ConvertFrom-Json
$dashboardEntry = Join-Path $dashboardRoot 'dashboard/app.py'
$dashboardListeners = @(Get-NetTCPConnection -LocalPort 8501 -State Listen -ErrorAction SilentlyContinue)
foreach ($dashboardOwner in ($dashboardListeners.OwningProcess | Select-Object -Unique)) {
    $dashboardProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $dashboardOwner"
    $dashboardOwnsProject = $dashboardProcess.CommandLine -like "*$dashboardEntry*"
    if (-not $dashboardOwnsProject -and $dashboardProcess.CommandLine -match 'streamlit\s+run\s+"?dashboard[/\\]app.py') {
        # Support a manually launched relative path only after checking its cwd.
        # If process inspection is unavailable, fail safely instead of guessing.
        & $dashboardRuntime.python_executable -c 'import pathlib,psutil,sys; sys.exit(0 if pathlib.Path(psutil.Process(int(sys.argv[1])).cwd()).resolve()==pathlib.Path(sys.argv[2]).resolve() else 1)' $dashboardOwner $dashboardRoot 2>$null
        $dashboardOwnsProject = $LASTEXITCODE -eq 0
    }
    if ($dashboardProcess.CommandLine -notmatch 'streamlit\s+run' -or -not $dashboardOwnsProject) {
        throw 'Port 8501 belongs to another service; no process was stopped.'
    }
    if ($dashboardProcess.CommandLine -notmatch '--server.port\s+8501') {
        throw 'Dashboard port could not be verified; no process was stopped.'
    }
    Stop-Process -Id $dashboardOwner -ErrorAction Stop
}
$dashboardLogs = Join-Path $dashboardRoot 'training_data/logs'
New-Item -ItemType Directory -Force -Path $dashboardLogs | Out-Null
Start-Process -FilePath $dashboardRuntime.python_executable -ArgumentList @('-m','streamlit','run',('"'+$dashboardEntry+'"'),'--server.address','0.0.0.0','--server.port','8501','--server.headless','true') -WorkingDirectory $dashboardRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $dashboardLogs 'dashboard-restart.out.log') -RedirectStandardError (Join-Path $dashboardLogs 'dashboard-restart.err.log')
for ($dashboardAttempt=0; $dashboardAttempt -lt 20; $dashboardAttempt++) {
    try {
        if ((Invoke-WebRequest 'http://127.0.0.1:8501/_stcore/health' -UseBasicParsing -TimeoutSec 2).Content.Trim() -eq 'ok') { exit 0 }
    } catch { }
    Start-Sleep -Milliseconds 500
}
throw 'Dashboard was launched, but its health check did not succeed.'
