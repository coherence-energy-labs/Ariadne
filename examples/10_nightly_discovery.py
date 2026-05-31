"""End-to-end demonstration of the nightly discovery operations loop.

Uses `synthetic_kepler` source so it runs without network access -- in production
you would set source='alerce_ztf' (or any other broker we add). The dedupe +
alert + history-tracking logic is identical regardless of source.

Run:  PYTHONPATH=src python examples/10_nightly_discovery.py
"""
import warnings
warnings.filterwarnings("ignore")

from ariadne.discovery.operations.nightly import NightlyConfig, run_nightly
from ariadne.discovery.operations.alerts import FileSink
from pathlib import Path

# Per-night config -- in production this would be loaded from a TOML/JSON file
# and the script invoked nightly by cron / systemd timer / Windows Task Scheduler.
cfg = NightlyConfig(
    store_path="data/discovery/candidates.json",
    source="synthetic_kepler",     # offline; switch to "alerce_ztf" for live data
    ra=180.0, dec=20.0, radius_deg=5.0,
    max_alerts=300,
    rms_threshold_arcsec=50.0,     # generous for synthetic Keplerian data
    do_xmatch=False,               # synthetic data has no SkyBoT analogue
    dry_run=False,
    alert_sinks=[FileSink("data/discovery/alerts.jsonl")],
)

print("Night 1: first run -- expect new candidates to be flagged + alerts fired")
summary = run_nightly(cfg)
print(f"  summary: {summary}\n")

print("Night 2: same config, re-run -- candidates already in store, NO new alerts")
summary2 = run_nightly(cfg)
print(f"  summary: {summary2}\n")

# Inspect what landed in the store
from ariadne.discovery.operations.candidate_store import CandidateStore
store = CandidateStore("data/discovery/candidates.json")
print(f"Final store: {len(store)} candidates total, "
      f"{len(store.discovery_candidates())} active discovery leads")
for c in list(store)[:5]:
    print(f"  {c.key}  ra={c.ra:.4f} dec={c.dec:+.4f}  rate={c.rate_arcsec_hr:.2f}\"/h  "
          f"first={c.first_seen_mjd:.1f}  n_runs={c.n_runs}  status={c.status}")

print(f"\nAlert audit log: data/discovery/alerts.jsonl")
log = Path("data/discovery/alerts.jsonl")
if log.exists():
    n_lines = sum(1 for _ in open(log))
    print(f"  {n_lines} alert records logged "
          f"(first run fired {summary['new']}, second fired {summary2['new']})")

print("\nProduction recipe:")
print("  Windows Task Scheduler: schedule 'ariadne nightly --store ... --source alerce_ztf ...'")
print("  Linux cron:             '0 12 * * * /path/to/venv/bin/ariadne nightly --store ...'")
print("  systemd timer:          unit + timer, OnCalendar=daily, runs the ariadne command")
