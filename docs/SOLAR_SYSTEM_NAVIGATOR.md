# Solar System Navigator

The solar-system navigator is the top-level route architect for missions beyond
the cislunar system. It combines:

- direct heliocentric Lambert porkchops on real ephemerides,
- global optimized gravity-assist chains,
- DSM-capable flyby scoring,
- Tisserand-style moon-tour screening for Jupiter and Saturn moons,
- Pareto ranking across time, delta-v, launch C3, arrival v-infinity, and risk,
- deterministic JSON route cards and PNG visual artifacts.

## CLI

```powershell
$env:PYTHONPATH="src"
python scripts\navigate_solar_system.py `
  --target Enceladus `
  --epoch-start 2028-01-01T00:00:00 `
  --departure-window-days 900 `
  --tof-min-days 700 `
  --tof-max-days 2400 `
  --n-dep 40 --n-tof 35 `
  --out-dir data\benchmarks\solar_navigator_enceladus
```

For expensive gravity-assist optimization:

```powershell
$env:PYTHONPATH="src"
python scripts\navigate_solar_system.py `
  --target Saturn `
  --epoch-start 2028-01-01T00:00:00 `
  --departure-window-days 1800 `
  --tof-min-days 900 `
  --tof-max-days 3200 `
  --optimize-flybys `
  --flyby-maxiter 80 `
  --out-dir data\benchmarks\solar_navigator_saturn_ga
```

## Python

```python
from ariadne.interplanetary.navigator import (
    NavigatorConstraints, navigate_solar_system, write_navigator_report)

report = navigate_solar_system(NavigatorConstraints(target="Titan"))
print(report.fastest)
print(report.cheapest)
print(report.balanced)
write_navigator_report(report, "data/benchmarks/solar_navigator_titan")
```

## Outputs

Each report includes:

- `navigator_report.json`: route cards with sequences, event times, C3,
  arrival v-infinity, component costs, assumptions, validations, and hashes.
- `route_cards.md`: human-review route cards for fastest, cheapest, balanced,
  Pareto routes, assumptions, and validation notes.
- `porkchop_heatmap.png`: direct Lambert cost heat map.
- `route_trade_space.png`: route delta-v/time trade plot with Pareto and
  balanced-route markings.

## Fidelity And Honesty

| Fidelity | Meaning |
|---|---|
| `patched_conic_ephemeris` | real ephemeris Lambert, patched-conic endpoint costs |
| `patched_conic_ephemeris+tisserand_screen` | adds moon-tour energy screening; not phased moon ephemerides |
| `optimized_flyby_chain` | globally optimized patched-conic flyby chain |

The navigator does not claim final flight readiness. It is a route-architecture
engine that says which families are promising, what assumptions they rely on,
and what should be promoted to high-fidelity targeting next.

## Current Real Smoke Result

Command:

```powershell
$env:PYTHONPATH="src"
python scripts\navigate_solar_system.py `
  --target Enceladus `
  --epoch-start 2028-01-01T00:00:00 `
  --departure-window-days 900 `
  --tof-min-days 700 `
  --tof-max-days 2400 `
  --n-dep 14 --n-tof 12 `
  --no-gravity-assist `
  --out-dir data\benchmarks\solar_navigator_enceladus_smoke
```

Result:

- balanced route: `direct_optimized_plus_saturn_moon_tour_to_enceladus`
- sequence: `EARTH -> SATURN BARYCENTER -> ENCELADUS`
- total proxy cost: `7305.36 m/s`
- TOF: `2168.70 days`
- C3: `106.51 km^2/s^2`
- arrival v-infinity: `5.75 km/s`
- certificate:
  `c302ace32f8741888dccb08e23c727785030d75fbd66012080a15cedea9f2395`

This is a direct-Lambert plus Saturn moon-tour screening result. The next step
for a mission-grade Enceladus study is to run the optimized flyby templates and
then promote the winning route to full multiple-shooting with Saturn-system
moon ephemerides.

## Benchmark And Validation Harness

Run:

```powershell
$env:PYTHONPATH="src"
python scripts\benchmark_solar_navigator.py `
  --out-dir data\benchmarks\solar_navigator_benchmark `
  --fail-on-error
```

The benchmark executes real-SPICE navigator cases, validates route-card
invariants, checks coarse physical sanity bounds, verifies generated PNGs, and
writes `benchmark_summary.json`.

Latest run:

- status: PASS
- elapsed: `0.31 s`
- benchmark certificate:
  `380762babc3ce1399f7ce41d162e42f8c5ba3fa6acc317dd3da1827c0835c810`
- Earth -> Mars balanced route: `3633.78 m/s`, C3 `9.18 km^2/s^2`,
  arrival v-infinity `2.72 km/s`, TOF `292.33 d`
- Earth -> Enceladus screening route: `7305.36 m/s`, C3
  `106.51 km^2/s^2`, arrival v-infinity `5.75 km/s`, TOF `2168.70 d`

The porkchop engine now reuses departure ephemeris states across each grid, so
large porkchop and launch-window sweeps avoid repeated SPICE calls for the same
departure epoch while preserving the public Lambert-transfer result exactly.

Navigator reports now retain representative direct-Pareto routes, not only the
single cheapest route. That exposes the real fastest/cheapest/balanced trade:
in the Enceladus smoke case the fastest screened route is 700 days at
`10993.16 m/s`, while the cheapest/balanced route is 2168.70 days at
`7305.36 m/s`.

To compare two benchmark summaries:

```powershell
$env:PYTHONPATH="src"
python scripts\compare_solar_navigator_benchmarks.py `
  baseline\benchmark_summary.json `
  candidate\benchmark_summary.json `
  --fail-on-drift
```

This reports changed cases, route-count deltas, balanced-route changes,
delta-v changes, and TOF changes.
