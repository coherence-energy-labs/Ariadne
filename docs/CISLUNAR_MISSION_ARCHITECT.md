# Cislunar Mission Architect

The mission architect ranks full Earth-Moon-Moon-Earth architectures by
combining Ariadne's direct ephemeris transfer, low-energy lunar capture,
coherence optimizer, and return-budget models into one auditable catalogue.

It is intentionally conservative: each leg carries a fidelity tag, assumptions,
validation notes, component delta-vs, robustness/risk scores, and a deterministic
certificate hash.

## Python

```python
import ariadne

report = ariadne.architect_cislunar_round_trip()
top = report.recommended
print(top.total_dv_ms, top.total_tof_days, top.outbound.name, top.return_leg.name)
```

## CLI

```powershell
$env:PYTHONPATH="src"
python scripts\architect_cislunar_mission.py `
  --out data\benchmarks\cislunar_architecture\report.json `
  --epoch 2025-06-01T00:00:00
```

For a fast ephemeris-only smoke run:

```powershell
$env:PYTHONPATH="src"
python scripts\architect_cislunar_mission.py `
  --out data\benchmarks\cislunar_architecture\direct_only.json `
  --outbound-tof-days 4.0 `
  --return-tof-days 3.0 `
  --no-low-energy --no-coherence
```

## Fidelity Tags

| Tag | Meaning |
|---|---|
| `analytic_patch` | patched-conic or analytic corridor proxy; useful for screening |
| `cr3bp_patch` | CR3BP/manifold result with patched external assumptions |
| `full_ephemeris` | full ephemeris propagation/targeting for the leg |
| `hybrid_verified` | Pareto-selected hybrid result with explicit robustness validation |

## What The Score Means

The scalar score is not a magic truth number. It is a ranking objective:

```text
score =
  w_dv * total_delta_v_km_s
  + w_tof * total_time_days
  + w_robustness * (1 - mean_robustness)
  + w_risk * mean_risk
  - w_fidelity_bonus * fidelity_rank
```

The Pareto frontier remains in the report so a reviewer can choose a route that
trades more time for lower delta-v, or more delta-v for robustness, without
trusting the scalar score blindly.

## Current Limit

Moon-to-Earth return is currently an analytic patched-corridor model, not yet a
full ephemeris shooting solve. That is deliberate: the architect marks this as
`analytic_patch` so the route card does not overclaim. The next upgrade is a
true full-ephemeris return optimizer with reentry corridor targeting.
