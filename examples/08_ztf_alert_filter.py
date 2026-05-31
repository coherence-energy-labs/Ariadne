"""ZTF alert -> moving-object candidate pipeline.

Pulls real ZTF alerts from ALeRCE for a small sky+time window, runs the full
Ariadne filter (cluster -> tracklet -> IOD+LM -> SkyBoT cross-match), and reports
candidate moving objects.

Honest scope:
- ~99% of alerts are real sub-transients (SNe, AGN, variable stars), not moving objects.
- The cone + time window chosen here is intentionally small (a few deg, a few nights)
  for tutorial speed. A nightly production run would do this for the whole survey area.
- A candidate that survives the pipeline (low-RMS orbit fit + no SkyBoT match) is
  worth following up. It is NOT YET a confirmed discovery -- that requires astrometric
  follow-up across multiple oppositions and submission to the MPC.

Run:  PYTHONPATH=src python examples/08_ztf_alert_filter.py
"""
import warnings, time
warnings.filterwarnings("ignore")

from ariadne.discovery.brokers.alerce import AlerceZTFBroker
from ariadne.discovery.brokers.base import collect
from ariadne.discovery import realtime

# Sky window: small box near ecliptic, well away from galactic plane / Moon.
# Default: a recent MJD window (last few weeks).
RA, DEC, RADIUS_DEG = 180.0, 0.0, 3.0
import datetime
MJD_END = (datetime.datetime.utcnow() - datetime.datetime(1858, 11, 17)).days
MJD_START = MJD_END - 14

print(f"Querying ALeRCE/ZTF for cone ({RA}, {DEC}) r={RADIUS_DEG} deg, MJD {MJD_START}..{MJD_END}")

broker = AlerceZTFBroker()
t0 = time.time()
alerts = collect(broker.query_cone(RA, DEC, RADIUS_DEG, MJD_START, MJD_END, max_alerts=2000),
                 max_n=2000)
print(f"  fetched {len(alerts)} alerts in {time.time()-t0:.1f}s\n")

if not alerts:
    print("No alerts returned. Try a different sky region, larger radius, or wider MJD window.")
    import sys; sys.exit(0)

result = realtime.run_pipeline(alerts,
                               cluster_pos_tol_arcsec=1.5,
                               rate_window_arcsec_hr=(0.05, 5.0),
                               rms_threshold_arcsec=15.0,
                               do_xmatch=True)

print("\nFINAL CANDIDATE REPORT")
print("-" * 76)
candidates = [t for t in result
              if t.get("status") == "accepted"
              and t.get("xmatch", {}).get("n_known") == 0]
known = [t for t in result
         if t.get("status") == "accepted"
         and t.get("xmatch", {}).get("n_known", 0) > 0]
print(f"  candidates (no SkyBoT match):  {len(candidates)}")
print(f"  re-detected known objects:     {len(known)}")
for k in known[:5]:
    print(f"    KNOWN: {k['xmatch']['names']} at "
          f"({k['ra']:.4f}, {k['dec']:.4f}) RMS={k['rms_arcsec']:.1f}\"")
for c in candidates[:10]:
    import math as _m
    print(f"    CANDIDATE: ({_m.degrees(c['ra']):.4f}, {_m.degrees(c['dec']):.4f}) "
          f"rate={c['rate_arcsec_hr']:.2f} as/hr  RMS={c['rms_arcsec']:.1f}\"")
print()
print("Honest scope: candidates require multi-opposition astrometric follow-up before")
print("submission to the MPC.  A single-night candidate is a lead, not a discovery.")
