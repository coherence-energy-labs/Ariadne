# Solar Navigator Route Cards: EARTH to ENCELADUS

- report certificate: `f89e79557d7cca09a1804f325c6c4bf47fbee7c8e8175737fb9418c62e3d9a40`
- routes: `9`
- fastest: `route_e90cb7ad9b86db`
- cheapest: `route_e560eba24bcf16`
- balanced: `route_e560eba24bcf16`

## Artifacts

- `mission_plate`: `data\benchmarks\solar_navigator_benchmark\earth_enceladus_small\mission_plate.png`
- `porkchop_heatmap`: `data\benchmarks\solar_navigator_benchmark\earth_enceladus_small\porkchop_heatmap.png`
- `report`: `data\benchmarks\solar_navigator_benchmark\earth_enceladus_small\navigator_report.json`
- `route_trade_space`: `data\benchmarks\solar_navigator_benchmark\earth_enceladus_small\route_trade_space.png`

## Summary

| Role | Route | Sequence | Cost m/s | TOF d | C3 | Arrival v-inf | Fidelity |
|---|---|---|---:|---:|---:|---:|---|
| fastest | `direct_pareto_00_plus_saturn_moon_tour_to_enceladus` | EARTH -> SATURN BARYCENTER -> ENCELADUS | 10993.16 | 700.00 | 231.402 | 18.730 | `patched_conic_ephemeris+tisserand_screen` |
| cheapest | `direct_optimized_plus_saturn_moon_tour_to_enceladus` | EARTH -> SATURN BARYCENTER -> ENCELADUS | 7305.36 | 2168.70 | 106.507 | 5.754 | `patched_conic_ephemeris+tisserand_screen` |
| balanced | `direct_optimized_plus_saturn_moon_tour_to_enceladus` | EARTH -> SATURN BARYCENTER -> ENCELADUS | 7305.36 | 2168.70 | 106.507 | 5.754 | `patched_conic_ephemeris+tisserand_screen` |

## Pareto Routes

| Route | Sequence | Cost m/s | TOF d | Risk | Certificate |
|---|---|---:|---:|---:|---|
| `direct_optimized_plus_saturn_moon_tour_to_enceladus` | EARTH -> SATURN BARYCENTER -> ENCELADUS | 7305.36 | 2168.70 | 0.492 | `e560eba24bcf169c` |
| `direct_pareto_05_plus_saturn_moon_tour_to_enceladus` | EARTH -> SATURN BARYCENTER -> ENCELADUS | 7654.80 | 1781.82 | 0.497 | `20375b979fd12638` |
| `direct_pareto_04_plus_saturn_moon_tour_to_enceladus` | EARTH -> SATURN BARYCENTER -> ENCELADUS | 7904.25 | 1318.18 | 0.565 | `f4af60ef3829cf25` |
| `direct_pareto_knee_plus_saturn_moon_tour_to_enceladus` | EARTH -> SATURN BARYCENTER -> ENCELADUS | 8080.23 | 1163.64 | 0.615 | `de254e1dd4fb7c47` |
| `direct_pareto_02_plus_saturn_moon_tour_to_enceladus` | EARTH -> SATURN BARYCENTER -> ENCELADUS | 8520.85 | 1009.09 | 0.686 | `d67fdd787f699bf3` |
| `direct_pareto_01_plus_saturn_moon_tour_to_enceladus` | EARTH -> SATURN BARYCENTER -> ENCELADUS | 9418.76 | 854.55 | 0.790 | `36860bbf402e9bb9` |
| `direct_pareto_00_plus_saturn_moon_tour_to_enceladus` | EARTH -> SATURN BARYCENTER -> ENCELADUS | 10993.16 | 700.00 | 0.924 | `e90cb7ad9b86db04` |

## Route Details

### direct_optimized_plus_saturn_moon_tour_to_enceladus

- id: `route_e560eba24bcf16`
- engine: `porkchop.optimize_window+tisserand_moon_tour`
- sequence: `EARTH -> SATURN BARYCENTER -> ENCELADUS`
- total cost: `7305.359 m/s`
- time of flight: `2168.703 d`
- C3: `106.50718367963891`
- arrival v-inf: `5.754144449978402`
- fidelity: `patched_conic_ephemeris+tisserand_screen`
- feasible: `True`
- Pareto: `True`
- certificate: `e560eba24bcf169c5ccaf82860fc5b502fa7093a53e0a0c5b8177b3647ef29c5`

Assumptions:
- heliocentric Lambert arc on real ephemerides
- departure parking-orbit cost used when configured; otherwise departure v-infinity
- capture cost included only for bodies with configured capture model
- moon tour is Tisserand energy screening, not phased moon ephemeris

Validations:
- launch C3 106.507 km^2/s^2
- arrival v_inf 5.754 km/s
- moon-tour deterministic mismatch 0.0 m/s
- moon-tour Hohmann baseline 0.0 m/s

### direct_pareto_05_plus_saturn_moon_tour_to_enceladus

- id: `route_20375b979fd126`
- engine: `porkchop.optimize_window+tisserand_moon_tour`
- sequence: `EARTH -> SATURN BARYCENTER -> ENCELADUS`
- total cost: `7654.800 m/s`
- time of flight: `1781.818 d`
- C3: `117.17517872603688`
- arrival v-inf: `5.911301908986419`
- fidelity: `patched_conic_ephemeris+tisserand_screen`
- feasible: `True`
- Pareto: `True`
- certificate: `20375b979fd1263819fa646cecd01d556812d0a1021dd43286579c83a3c6f9ea`

Assumptions:
- heliocentric Lambert arc on real ephemerides
- departure parking-orbit cost used when configured; otherwise departure v-infinity
- capture cost included only for bodies with configured capture model
- moon tour is Tisserand energy screening, not phased moon ephemeris

Validations:
- launch C3 117.175 km^2/s^2
- arrival v_inf 5.911 km/s
- moon-tour deterministic mismatch 0.0 m/s
- moon-tour Hohmann baseline 0.0 m/s

### direct_pareto_04_plus_saturn_moon_tour_to_enceladus

- id: `route_f4af60ef3829cf`
- engine: `porkchop.optimize_window+tisserand_moon_tour`
- sequence: `EARTH -> SATURN BARYCENTER -> ENCELADUS`
- total cost: `7904.252 m/s`
- time of flight: `1318.182 d`
- C3: `124.94000133166263`
- arrival v-inf: `7.937491303472553`
- fidelity: `patched_conic_ephemeris+tisserand_screen`
- feasible: `True`
- Pareto: `True`
- certificate: `f4af60ef3829cf258c1a7bf4e6fb29a59e34e43374d107a814b29b8337f472aa`

Assumptions:
- heliocentric Lambert arc on real ephemerides
- departure parking-orbit cost used when configured; otherwise departure v-infinity
- capture cost included only for bodies with configured capture model
- moon tour is Tisserand energy screening, not phased moon ephemeris

Validations:
- launch C3 124.940 km^2/s^2
- arrival v_inf 7.937 km/s
- moon-tour deterministic mismatch 0.0 m/s
- moon-tour Hohmann baseline 0.0 m/s

### direct_pareto_knee_plus_saturn_moon_tour_to_enceladus

- id: `route_de254e1dd4fb7c`
- engine: `porkchop.optimize_window+tisserand_moon_tour`
- sequence: `EARTH -> SATURN BARYCENTER -> ENCELADUS`
- total cost: `8080.234 m/s`
- time of flight: `1163.636 d`
- C3: `130.4927693357062`
- arrival v-inf: `9.437556414102703`
- fidelity: `patched_conic_ephemeris+tisserand_screen`
- feasible: `True`
- Pareto: `True`
- certificate: `de254e1dd4fb7c474a66f7f13e47098cf781212a522cec1f5768956dbb59c6f9`

Assumptions:
- heliocentric Lambert arc on real ephemerides
- departure parking-orbit cost used when configured; otherwise departure v-infinity
- capture cost included only for bodies with configured capture model
- moon tour is Tisserand energy screening, not phased moon ephemeris

Validations:
- launch C3 130.493 km^2/s^2
- arrival v_inf 9.438 km/s
- moon-tour deterministic mismatch 0.0 m/s
- moon-tour Hohmann baseline 0.0 m/s

### direct_grid_best_plus_saturn_moon_tour_to_enceladus

- id: `route_d85d2239326f4c`
- engine: `porkchop.optimize_window+tisserand_moon_tour`
- sequence: `EARTH -> SATURN BARYCENTER -> ENCELADUS`
- total cost: `7577.784 m/s`
- time of flight: `2245.455 d`
- C3: `114.8029740769316`
- arrival v-inf: `5.832583436247178`
- fidelity: `patched_conic_ephemeris+tisserand_screen`
- feasible: `True`
- Pareto: `False`
- certificate: `d85d2239326f4c7a5d6dc5359fceccda115555982ecff782827d5544e605dab4`

Assumptions:
- heliocentric Lambert arc on real ephemerides
- departure parking-orbit cost used when configured; otherwise departure v-infinity
- capture cost included only for bodies with configured capture model
- moon tour is Tisserand energy screening, not phased moon ephemeris

Validations:
- launch C3 114.803 km^2/s^2
- arrival v_inf 5.833 km/s
- moon-tour deterministic mismatch 0.0 m/s
- moon-tour Hohmann baseline 0.0 m/s

### direct_pareto_06_plus_saturn_moon_tour_to_enceladus

- id: `route_b87e83fe7b829c`
- engine: `porkchop.optimize_window+tisserand_moon_tour`
- sequence: `EARTH -> SATURN BARYCENTER -> ENCELADUS`
- total cost: `7577.784 m/s`
- time of flight: `2245.455 d`
- C3: `114.8029740769316`
- arrival v-inf: `5.832583436247178`
- fidelity: `patched_conic_ephemeris+tisserand_screen`
- feasible: `True`
- Pareto: `False`
- certificate: `b87e83fe7b829c634668d2aa5bd55a13eb6d24b276b8d9bb781c05ab3a1669de`

Assumptions:
- heliocentric Lambert arc on real ephemerides
- departure parking-orbit cost used when configured; otherwise departure v-infinity
- capture cost included only for bodies with configured capture model
- moon tour is Tisserand energy screening, not phased moon ephemeris

Validations:
- launch C3 114.803 km^2/s^2
- arrival v_inf 5.833 km/s
- moon-tour deterministic mismatch 0.0 m/s
- moon-tour Hohmann baseline 0.0 m/s

### direct_pareto_02_plus_saturn_moon_tour_to_enceladus

- id: `route_d67fdd787f699b`
- engine: `porkchop.optimize_window+tisserand_moon_tour`
- sequence: `EARTH -> SATURN BARYCENTER -> ENCELADUS`
- total cost: `8520.850 m/s`
- time of flight: `1009.091 d`
- C3: `144.66721200918113`
- arrival v-inf: `11.591934180670922`
- fidelity: `patched_conic_ephemeris+tisserand_screen`
- feasible: `True`
- Pareto: `True`
- certificate: `d67fdd787f699bf3100fa8bcf144c5c1803a0145a9d18c6354f527cc29c49d79`

Assumptions:
- heliocentric Lambert arc on real ephemerides
- departure parking-orbit cost used when configured; otherwise departure v-infinity
- capture cost included only for bodies with configured capture model
- moon tour is Tisserand energy screening, not phased moon ephemeris

Validations:
- launch C3 144.667 km^2/s^2
- arrival v_inf 11.592 km/s
- moon-tour deterministic mismatch 0.0 m/s
- moon-tour Hohmann baseline 0.0 m/s

### direct_pareto_01_plus_saturn_moon_tour_to_enceladus

- id: `route_36860bbf402e9b`
- engine: `porkchop.optimize_window+tisserand_moon_tour`
- sequence: `EARTH -> SATURN BARYCENTER -> ENCELADUS`
- total cost: `9418.761 m/s`
- time of flight: `854.545 d`
- C3: `174.7545217098506`
- arrival v-inf: `14.686979258633258`
- fidelity: `patched_conic_ephemeris+tisserand_screen`
- feasible: `True`
- Pareto: `True`
- certificate: `36860bbf402e9bb916934650e3689c16e22ae5fb2d8e9f110aa4bc33eee704ce`

Assumptions:
- heliocentric Lambert arc on real ephemerides
- departure parking-orbit cost used when configured; otherwise departure v-infinity
- capture cost included only for bodies with configured capture model
- moon tour is Tisserand energy screening, not phased moon ephemeris

Validations:
- launch C3 174.755 km^2/s^2
- arrival v_inf 14.687 km/s
- moon-tour deterministic mismatch 0.0 m/s
- moon-tour Hohmann baseline 0.0 m/s

### direct_pareto_00_plus_saturn_moon_tour_to_enceladus

- id: `route_e90cb7ad9b86db`
- engine: `porkchop.optimize_window+tisserand_moon_tour`
- sequence: `EARTH -> SATURN BARYCENTER -> ENCELADUS`
- total cost: `10993.156 m/s`
- time of flight: `700.000 d`
- C3: `231.40196863699038`
- arrival v-inf: `18.73036198664966`
- fidelity: `patched_conic_ephemeris+tisserand_screen`
- feasible: `True`
- Pareto: `True`
- certificate: `e90cb7ad9b86db044b50ec0ea9fc56eaf74afaeee4a01801dde7ad3a140bb38e`

Assumptions:
- heliocentric Lambert arc on real ephemerides
- departure parking-orbit cost used when configured; otherwise departure v-infinity
- capture cost included only for bodies with configured capture model
- moon tour is Tisserand energy screening, not phased moon ephemeris

Validations:
- launch C3 231.402 km^2/s^2
- arrival v_inf 18.730 km/s
- moon-tour deterministic mismatch 0.0 m/s
- moon-tour Hohmann baseline 0.0 m/s
