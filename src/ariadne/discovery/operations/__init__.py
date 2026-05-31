"""Discovery operations -- the loop that turns the validated engine into a
running discovery service.

Components:
  - candidate_store.py : JSON-backed persistence of candidate orbits across runs.
                         Dedupes by sky-position + rate; tracks first-seen,
                         last-seen, n-observations, RMS history per candidate.
  - alerts.py          : pluggable notification sinks -- log file, webhook
                         (Slack/Discord/generic), email -- for new candidates
                         that surface in a nightly run.
  - nightly.py         : the orchestrator. One CLI invocation: pull a sky
                         window's alerts/detections, run the discovery pipeline,
                         dedupe vs the candidate store, fire alerts on new
                         candidates, append the run history. Idempotent under
                         re-invocation (won't re-alert the same candidate).

Designed to run as a scheduled job (cron / systemd timer / Windows Task
Scheduler). One config file + one invocation per night.
"""
from .candidate_store import Candidate, CandidateStore
from .alerts import AlertSink, NoopSink, FileSink, WebhookSink, EmailSink

__all__ = [
    "Candidate", "CandidateStore",
    "AlertSink", "NoopSink", "FileSink", "WebhookSink", "EmailSink",
]
