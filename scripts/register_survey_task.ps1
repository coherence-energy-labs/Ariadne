# Register (or remove) the Ariadne automated-discovery Windows Scheduled Task.
#   Register:  powershell -ExecutionPolicy Bypass -File scripts\register_survey_task.ps1
#   Remove:    powershell -ExecutionPolicy Bypass -File scripts\register_survey_task.ps1 -Remove
# The task runs run_auto_survey.ps1 once a day (machine must be powered on), which
# fetches one ecliptic target field, runs discovery, and updates DISCOVERY_WATCH.md.
param([switch]$Remove)

$TaskName = "AriadneAutoDiscovery"
$Root = Split-Path -Parent $PSScriptRoot
$Survey = Join-Path $Root "scripts\run_auto_survey.ps1"

if ($Remove) {
  schtasks /Delete /TN $TaskName /F
  Write-Output "Removed scheduled task '$TaskName'."
  exit 0
}

$action = "powershell -NoProfile -ExecutionPolicy Bypass -File `"$Survey`""
# daily at 02:47 local (off-peak, off-minute); machine must be on at that time.
schtasks /Create /TN $TaskName /TR $action /SC DAILY /ST 02:47 /F
Write-Output "Registered '$TaskName' (daily 02:47). Survey: $Survey"
Write-Output "Run once now to test:  schtasks /Run /TN $TaskName"
Write-Output "Remove:                powershell -ExecutionPolicy Bypass -File scripts\register_survey_task.ps1 -Remove"
