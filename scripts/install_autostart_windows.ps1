# Install Windows auto-start: bot resumes after PC reboot / login.
# Run once in PowerShell (can be non-admin for current-user task):
#   powershell -ExecutionPolicy Bypass -File scripts\install_autostart_windows.ps1

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Bat = Join-Path $Root "scripts\start_bot.bat"
$TaskName = "TryzubTradeBot"

if (-not (Test-Path $Bat)) {
  Write-Error "Missing $Bat"
}

# Remove old task if present
schtasks /Query /TN $TaskName 2>$null | Out-Null
if ($LASTEXITCODE -eq 0) {
  schtasks /Delete /TN $TaskName /F | Out-Null
}

# Start at user logon, restart on failure
$tr = "`"$Bat`""
schtasks /Create /TN $TaskName /TR $tr /SC ONLOGON /RL LIMITED /F | Out-Null

Write-Host "OK: scheduled task '$TaskName' created."
Write-Host "The bot will start automatically when you log into Windows."
Write-Host "It continues from saved DB + current brain (does not retrain from zero)."
Write-Host "Manual start: $Bat"
Write-Host "Disable: schtasks /Delete /TN $TaskName /F"
