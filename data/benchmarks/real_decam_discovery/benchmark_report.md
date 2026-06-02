# Real DECam Discovery Benchmark

Generated: 2026-06-02

## Configuration

- Field: cached multi-night DECam near-ecliptic discovery field
- Usable exposures: 15 / 18
- Truth catalog scan: MPC known-object database, 1,600,000-row candidate limit
- Truth propagation: N-body ephemeris
- Truth match radius: 5 arcsec
- Link detections: uncapped per exposure, stationary-source veto at 1.2 arcsec
- Tracklet cap: 3,000 per night
- Candidate vetting: 4-night minimum, rate CV <= 0.20, heading scatter <= 18 deg, linear residual <= 30 arcsec

## Measured Improvement

| Metric | Earlier Real-Field Run | Current Run |
|---|---:|---:|
| Known recoverable objects | 129 | 135 |
| Known objects recovered | 0 | 21 |
| Known recovery | 0.000 | 0.156 |
| Runtime | 639.8 s | 254.5 s |
| Tracklet build time | 563.4 s | 69.3 s |
| Link detections | 175,319 | 88,282 |
| Tracklets | 2,000 | 2,252 |
| Unknown multi-night chains | 225 | 285 |
| Vetted unknown candidates | 0 | 1 |
| Scrambled false floor | 216 | 271 |
| Vetted candidate / false-floor ratio | 0.000 | 0.00369 |

## Interpretation

The previous real-field score was not an engine ceiling. It was dominated by fixed-source combinatorics and by truth labels that used night-median ephemerides. The current benchmark uses per-exposure truth labels and removes same-night fixed-source repeats before image tracklet formation.

The result is a real measured recovery improvement on labelled external data: 21 known multi-night objects recovered where the prior run recovered none, while reducing total runtime by 60.2% and tracklet build time by 87.7%.

The remaining recall gap is now localized to the image discovery front end: source extraction depth, trailed/moving-source detection, and cross-night linking tolerance. The benchmark is now capable of seeing those improvements instead of reporting a misleading zero.
