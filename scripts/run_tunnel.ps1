$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$runtimePath = Join-Path $projectRoot "training_data\automation\runtime.json"
$logDirectory = Join-Path $projectRoot "training_data\logs"
$logPath = Join-Path $logDirectory "tunnel.log"

if (-not (Test-Path -LiteralPath $runtimePath)) {
    throw "Automation runtime file is missing."
}

$runtime = Get-Content -Raw -LiteralPath $runtimePath | ConvertFrom-Json
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
if ((Test-Path -LiteralPath $logPath) -and (Get-Item -LiteralPath $logPath).Length -gt 5MB) {
    Move-Item -Force -LiteralPath $logPath -Destination "$logPath.1"
}
Set-Location -LiteralPath $runtime.project_root

& $runtime.ngrok_executable http 8501 --config $runtime.ngrok_config --traffic-policy-file $runtime.ngrok_policy --log $logPath --log-format json
$tunnelExitCode = $LASTEXITCODE
if (Test-Path -LiteralPath $logPath) {
    $sanitized = (Get-Content -Raw -LiteralPath $logPath) -replace 'Your authtoken: [^\\r\\n]+', 'Your authtoken: [REDACTED]'
    Set-Content -LiteralPath $logPath -Value $sanitized -Encoding UTF8
}
exit $tunnelExitCode
