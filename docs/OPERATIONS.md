# Ariadne discovery operations — running the engine as a daily service

The discovery pipeline is designed to be run as a scheduled job. Components:

- **Candidate store** (`ariadne.discovery.operations.candidate_store`): JSON-backed
  persistence with dedupe by sky-position + rate. New candidates surface in alerts
  the first time only; re-detections update history without re-firing.
- **Alert sinks** (`ariadne.discovery.operations.alerts`): pluggable notifications.
  `FileSink` (JSON-Lines audit log), `WebhookSink` (Slack/Discord/generic),
  `EmailSink` (SMTP).
- **Nightly orchestrator** (`ariadne.discovery.operations.nightly`): one CLI =
  one night = pull → filter → dedupe → fire-alerts → save store.

## CLI invocation

```bash
ariadne nightly \
  --store     /var/lib/ariadne/candidates.json \
  --source    alerce_ztf \
  --ra        180.0 --dec 20.0 --radius 5.0 \
  --max-alerts 2000 \
  --rms-threshold 15.0 \
  --alert-log /var/log/ariadne/alerts.jsonl \
  --webhook-url $ARIADNE_SLACK_WEBHOOK
```

The candidate store and alert log are append-only; safe to inspect mid-run.

## Scheduling

### Linux — systemd timer (recommended)

`/etc/systemd/system/ariadne-nightly.service`:
```ini
[Unit]
Description=Ariadne nightly TNO discovery
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
User=ariadne
WorkingDirectory=/opt/ariadne
EnvironmentFile=/etc/ariadne/env
ExecStart=/opt/ariadne/.venv/bin/ariadne nightly \
    --store /var/lib/ariadne/candidates.json \
    --source alerce_ztf --ra 180 --dec 20 --radius 5 \
    --alert-log /var/log/ariadne/alerts.jsonl \
    --webhook-url ${ARIADNE_SLACK_WEBHOOK}
```

`/etc/systemd/system/ariadne-nightly.timer`:
```ini
[Unit]
Description=Run Ariadne discovery every day at 12:00 UTC

[Timer]
OnCalendar=*-*-* 12:00:00
Persistent=true

[Install]
WantedBy=timers.target
```

Activate: `systemctl --user enable --now ariadne-nightly.timer`.

### Linux — classic cron

`/etc/cron.d/ariadne`:
```
0 12 * * * ariadne /opt/ariadne/.venv/bin/ariadne nightly \
  --store /var/lib/ariadne/candidates.json --source alerce_ztf \
  --ra 180 --dec 20 --radius 5 \
  --alert-log /var/log/ariadne/alerts.jsonl \
  --webhook-url ${ARIADNE_SLACK_WEBHOOK} 2>>/var/log/ariadne/cron.log
```

### Windows — Task Scheduler

PowerShell (run as admin):
```powershell
$action  = New-ScheduledTaskAction -Execute "C:\Path\To\python.exe" `
  -Argument "-m ariadne.cli nightly --store C:\ariadne\candidates.json `
            --source alerce_ztf --ra 180 --dec 20 --radius 5 `
            --alert-log C:\ariadne\alerts.jsonl"
$trigger = New-ScheduledTaskTrigger -Daily -At 5am
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable
Register-ScheduledTask -TaskName "Ariadne nightly" -Action $action `
  -Trigger $trigger -Settings $settings -Description "Ariadne TNO discovery"
```

### Docker (any host)

`Dockerfile`:
```dockerfile
FROM python:3.12-slim
RUN pip install ariadne-astro
ENTRYPOINT ["ariadne", "nightly"]
```

Run via cron / systemd / Kubernetes CronJob on the host.

## Configuration patterns

### Multiple sky patches

One config = one cone. To survey several patches per night, run the command
several times with different `--ra/--dec` and the SAME `--store`. The store
dedupes across runs.

### Email alerts (requires SMTP)

The CLI doesn't expose `EmailSink` directly because SMTP credentials shouldn't
land on the command line. Use the Python API:

```python
import os
from ariadne.discovery.operations.nightly import NightlyConfig, run_nightly
from ariadne.discovery.operations.alerts import (
    FileSink, EmailSink, SMTPConfig)

cfg = NightlyConfig(
    store_path="data/candidates.json",
    source="alerce_ztf",
    ra=180, dec=20, radius_deg=5,
    alert_sinks=[
        FileSink("data/alerts.jsonl"),
        EmailSink(
            smtp=SMTPConfig(
                host="smtp.example.com", port=587,
                username=os.environ["ARIADNE_SMTP_USER"],
                password=os.environ["ARIADNE_SMTP_PASS"],
            ),
            sender="ariadne@example.com",
            recipients=["you@example.com"],
        ),
    ],
)
run_nightly(cfg)
```

## Monitoring

- `--alert-log` is JSON-Lines: easy to tail / grep / pipe.
- Each run prints a one-line summary (`alerts`, `accepted`, `new`, `stale_marked`, `elapsed_s`).
- Inspect the candidate store at any time:
  ```python
  from ariadne.discovery.operations.candidate_store import CandidateStore
  s = CandidateStore("data/candidates.json")
  for c in s.discovery_candidates():
      print(c)
  ```

## What "discovery candidate" means

A candidate that:
1. Survived the IOD+LM orbit-fit filter (RMS below threshold), AND
2. Cross-matched against SkyBoT and returned NO known object within tolerance.

**This is a LEAD, not a discovery.** Multi-opposition follow-up is required to
submit to the MPC. The recommended next step on any candidate is to image the
field again at the next opposition (~6 months later) and confirm the orbit
predicts the new detection.

## Honest scope

- The pipeline is bounded by upstream-data density. ALeRCE alone delivers
  mostly single-detection objects — not enough to build orbit-fittable
  tracklets. To find new things, use the **image-based** pipeline
  (`discovery/imaging`) against deeper-than-MPC archives (DECam DR10, Subaru
  HSC, PanSTARRS warps).
- Nightly runs are *cheap* on synthetic data and *expensive* on archive-image
  paths (each DECam tile is 30–200 MB; full nightly sweeps need parallelisation).
- Operational discovery work needs human review of every candidate flagged.
  Ariadne provides the filter; final calls are yours.
