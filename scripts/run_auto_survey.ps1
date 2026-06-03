# Ariadne automated survey loop -- one invocation per scheduled run.
# Rotates through data/survey_targets.txt, fetches a fresh multi-night DECam field
# from the NOIRLab archive, runs the discovery orchestrator on it (which dedups +
# persists candidates + updates DISCOVERY_WATCH.md), then cleans up the FITS.
# Register with Windows Task Scheduler to run unattended (see register_survey_task.ps1).

$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot
$Targets = Join-Path $Root "data\survey_targets.txt"
$Log = Join-Path $Root "data\auto_survey.log"
$Py = "python"

function Log($msg) {
  $line = "$(Get-Date -Format o)  $msg"
  Add-Content -Path $Log -Value $line -Encoding utf8
  Write-Output $line
}

# parse targets (skip comments/blanks)
$rows = @()
foreach ($ln in Get-Content $Targets) {
  if ($ln.Trim() -eq "" -or $ln.Trim().StartsWith("#")) { continue }
  $p = $ln -split "\s+"
  if ($p.Count -ge 3) { $rows += ,@([double]$p[0], [double]$p[1], $p[2]) }
}
if ($rows.Count -eq 0) { Log "no targets"; exit 1 }

# pick the target for today (rotate by day-of-year so it sweeps the ecliptic)
$idx = ([int](Get-Date).DayOfYear) % $rows.Count
$t = $rows[$idx]
$ra = $t[0]; $dec = $t[1]; $label = $t[2]
Log "survey run: target $label  RA=$ra Dec=$dec (idx $idx / $($rows.Count))"

# fetch + process + cleanup. The orchestrator exits non-zero if too few exposures
# (e.g. no recent NOIRLab coverage there) -- that is fine, just log and move on.
& $Py (Join-Path $Root "scripts\run_auto_discovery.py") `
    --fetch --ra $ra --dec $dec --field-id $label --nights 4 --cleanup 2>&1 |
  ForEach-Object { Log $_ }

# CATALOG-MODE sweep: query the NSC detection catalog for a batch of ecliptic
# tiles (NO image download). Productive only with Data Lab credentials
# (COH_DATALAB_TOKEN or DATALAB_USER/PASS); otherwise it reports + no-ops.
Log "catalog sweep (NSC tiles, no images)"
& $Py (Join-Path $Root "scripts\run_catalog_survey.py") --tiles-per-run 15 --radius 0.12 2>&1 |
  ForEach-Object { Log $_ }

Log "survey run complete for $label"
