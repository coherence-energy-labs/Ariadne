"""Alert sinks -- pluggable notifications when a new discovery candidate surfaces.

Sinks (one or more, chained):
  - NoopSink:    discard alerts (default, useful for dry-runs)
  - FileSink:    append a one-line JSON-Lines record per alert (audit log)
  - WebhookSink: POST to Slack / Discord / generic JSON webhook
  - EmailSink:   send SMTP email (requires SMTP server config)

A nightly run typically uses FileSink + one of WebhookSink / EmailSink.
"""
from __future__ import annotations

import json
import smtplib
import time
from dataclasses import dataclass
from email.mime.text import MIMEText
from pathlib import Path

from .candidate_store import Candidate


def _format_alert_text(c: Candidate, run_id: str | None = None) -> str:
    """Compact human-readable alert text."""
    age_days = c.last_seen_mjd - c.first_seen_mjd if c.n_runs > 1 else 0
    rms = c.rms_history[-1][1] if c.rms_history else float("nan")
    lines = [
        f"Ariadne candidate {c.key}",
        f"  position : ({c.ra:.4f}, {c.dec:+.4f}) deg",
        f"  rate     : {c.rate_arcsec_hr:.2f} arcsec/hr",
        f"  RMS      : {rms:.1f} arcsec",
        f"  first    : MJD {c.first_seen_mjd:.3f}",
        f"  last     : MJD {c.last_seen_mjd:.3f} ({c.n_runs} runs, {age_days:.1f} d arc)",
        f"  status   : {c.status}",
        f"  skybot   : {c.skybot_names if c.skybot_names else '(no known match)'}",
    ]
    if run_id:
        lines.insert(1, f"  run      : {run_id}")
    return "\n".join(lines)


class AlertSink:
    """Base class. Subclasses implement `fire(candidate, run_id=None)`."""

    name = "noop"

    def fire(self, candidate: Candidate, run_id: str | None = None) -> None:
        raise NotImplementedError


class NoopSink(AlertSink):
    """Discard. Useful for dry-runs."""
    name = "noop"

    def fire(self, candidate, run_id=None):
        pass


class FileSink(AlertSink):
    """JSON-Lines audit log -- one record per alert.

    Each line: {ts, run_id, key, ra, dec, rate, rms, status, skybot, n_runs}
    """
    name = "file"

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def fire(self, candidate, run_id=None):
        rec = {
            "ts": time.time(),
            "run_id": run_id,
            "key": candidate.key,
            "ra": candidate.ra,
            "dec": candidate.dec,
            "rate_arcsec_hr": candidate.rate_arcsec_hr,
            "rms_arcsec": candidate.rms_history[-1][1] if candidate.rms_history else None,
            "status": candidate.status,
            "skybot_names": candidate.skybot_names,
            "n_runs": candidate.n_runs,
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")


class WebhookSink(AlertSink):
    """Generic JSON webhook -- works with Slack, Discord, or any HTTPS endpoint.

    For Slack: format= "slack" sends `{"text": "..."}`.
    For Discord: format= "discord" sends `{"content": "..."}`.
    For raw:   format= "raw" sends the full alert dict.
    """
    name = "webhook"

    def __init__(self, url: str, format: str = "slack", timeout: float = 10.0):
        if not url.startswith("https://") and not url.startswith("http://"):
            raise ValueError("WebhookSink: url must start with http:// or https://")
        self.url = url
        self.format = format
        self.timeout = timeout

    def fire(self, candidate, run_id=None):
        import requests
        msg = _format_alert_text(candidate, run_id)
        if self.format == "slack":
            payload = {"text": "```\n" + msg + "\n```"}
        elif self.format == "discord":
            payload = {"content": "```\n" + msg + "\n```"}
        else:
            payload = {
                "text": msg, "run_id": run_id,
                "candidate": {
                    "key": candidate.key, "ra": candidate.ra, "dec": candidate.dec,
                    "rate_arcsec_hr": candidate.rate_arcsec_hr,
                    "rms_arcsec": candidate.rms_history[-1][1] if candidate.rms_history else None,
                    "n_runs": candidate.n_runs, "status": candidate.status,
                    "skybot_names": candidate.skybot_names,
                },
            }
        try:
            requests.post(self.url, json=payload, timeout=self.timeout)
        except Exception as e:
            # log to stderr but don't kill the nightly run
            import sys
            print(f"WebhookSink: POST failed: {e}", file=sys.stderr)


@dataclass
class SMTPConfig:
    """SMTP server config for EmailSink."""
    host: str
    port: int = 587
    username: str = ""
    password: str = ""
    use_tls: bool = True


class EmailSink(AlertSink):
    """SMTP email alert (requires SMTPConfig + working SMTP server).

    Pass credentials via env vars + read in your nightly script -- DO NOT hardcode
    credentials in the script.
    """
    name = "email"

    def __init__(self, smtp: SMTPConfig, sender: str, recipients: list[str],
                 subject_prefix: str = "[Ariadne candidate] "):
        self.smtp = smtp
        self.sender = sender
        self.recipients = recipients
        self.subject_prefix = subject_prefix

    def fire(self, candidate, run_id=None):
        msg = MIMEText(_format_alert_text(candidate, run_id))
        msg["Subject"] = (self.subject_prefix +
                          f"{candidate.key} (RMS {candidate.rms_history[-1][1]:.1f}\")"
                          if candidate.rms_history else self.subject_prefix + candidate.key)
        msg["From"] = self.sender
        msg["To"] = ", ".join(self.recipients)
        try:
            with smtplib.SMTP(self.smtp.host, self.smtp.port) as s:
                if self.smtp.use_tls:
                    s.starttls()
                if self.smtp.username and self.smtp.password:
                    s.login(self.smtp.username, self.smtp.password)
                s.send_message(msg)
        except Exception as e:
            import sys
            print(f"EmailSink: SMTP send failed: {e}", file=sys.stderr)


def fire_all(sinks: list[AlertSink], candidate: Candidate, run_id: str | None = None):
    """Helper: fire one candidate to every sink, ignoring per-sink errors."""
    for sink in sinks:
        try:
            sink.fire(candidate, run_id=run_id)
        except Exception as e:
            import sys
            print(f"{sink.name} sink failed: {e}", file=sys.stderr)
