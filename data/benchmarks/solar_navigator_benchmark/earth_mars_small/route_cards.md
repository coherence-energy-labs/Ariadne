# Solar Navigator Route Cards: EARTH to MARS

- report certificate: `c622683a3d7bbc94c0a3f8aff4c94fa345269ed6610e5c5508d98c4041050567`
- routes: `12`
- fastest: `route_12a8b5753f0a5a`
- cheapest: `route_360cab946c0202`
- balanced: `route_360cab946c0202`

## Artifacts

- `mission_plate`: `data\benchmarks\solar_navigator_benchmark\earth_mars_small\mission_plate.png`
- `porkchop_heatmap`: `data\benchmarks\solar_navigator_benchmark\earth_mars_small\porkchop_heatmap.png`
- `report`: `data\benchmarks\solar_navigator_benchmark\earth_mars_small\navigator_report.json`
- `route_trade_space`: `data\benchmarks\solar_navigator_benchmark\earth_mars_small\route_trade_space.png`

## Summary

| Role | Route | Sequence | Cost m/s | TOF d | C3 | Arrival v-inf | Fidelity |
|---|---|---|---:|---:|---:|---:|---|
| fastest | `direct_pareto_00` | EARTH -> MARS BARYCENTER | 5214.30 | 120.00 | 47.773 | 11.141 | `patched_conic_ephemeris` |
| cheapest | `direct_optimized` | EARTH -> MARS BARYCENTER | 3633.78 | 292.33 | 9.182 | 2.716 | `patched_conic_ephemeris` |
| balanced | `direct_optimized` | EARTH -> MARS BARYCENTER | 3633.78 | 292.33 | 9.182 | 2.716 | `patched_conic_ephemeris` |

## Pareto Routes

| Route | Sequence | Cost m/s | TOF d | Risk | Certificate |
|---|---|---:|---:|---:|---|
| `direct_optimized` | EARTH -> MARS BARYCENTER | 3633.78 | 292.33 | 0.271 | `360cab946c020238` |
| `direct_pareto_06` | EARTH -> MARS BARYCENTER | 3746.27 | 258.46 | 0.284 | `55443dabbedea8ed` |
| `direct_pareto_05` | EARTH -> MARS BARYCENTER | 3780.98 | 235.38 | 0.310 | `0d2b2c5634c4c1d4` |
| `direct_pareto_04` | EARTH -> MARS BARYCENTER | 3830.30 | 212.31 | 0.348 | `ea9b540581971395` |
| `direct_pareto_knee` | EARTH -> MARS BARYCENTER | 3945.41 | 189.23 | 0.399 | `3a1673f6b2eac363` |
| `direct_pareto_02` | EARTH -> MARS BARYCENTER | 4218.57 | 166.15 | 0.465 | `da892619bea14d08` |
| `direct_pareto_01` | EARTH -> MARS BARYCENTER | 4831.58 | 143.08 | 0.554 | `4f2e5702363048a4` |
| `direct_pareto_00` | EARTH -> MARS BARYCENTER | 5214.30 | 120.00 | 0.551 | `12a8b5753f0a5acb` |

## Route Details

### direct_optimized

- id: `route_360cab946c0202`
- engine: `porkchop.optimize_window`
- sequence: `EARTH -> MARS BARYCENTER`
- total cost: `3633.777 m/s`
- time of flight: `292.327 d`
- C3: `9.18214113116599`
- arrival v-inf: `2.716245869853926`
- fidelity: `patched_conic_ephemeris`
- feasible: `True`
- Pareto: `True`
- certificate: `360cab946c0202388082dee3ac69be8d8929506dce1789dc971fb1abe2e42aa2`

Assumptions:
- heliocentric Lambert arc on real ephemerides
- departure parking-orbit cost used when configured; otherwise departure v-infinity
- capture cost included only for bodies with configured capture model

Validations:
- launch C3 9.182 km^2/s^2
- arrival v_inf 2.716 km/s

### direct_pareto_07

- id: `route_182991ad071d1b`
- engine: `porkchop.optimize_window`
- sequence: `EARTH -> MARS BARYCENTER`
- total cost: `3716.636 m/s`
- time of flight: `304.615 d`
- C3: `11.081177034723648`
- arrival v-inf: `2.778148243111514`
- fidelity: `patched_conic_ephemeris`
- feasible: `True`
- Pareto: `False`
- certificate: `182991ad071d1babdd089371075a73bfef97be74d298724a7e0cf8a1e7f5765c`

Assumptions:
- heliocentric Lambert arc on real ephemerides
- departure parking-orbit cost used when configured; otherwise departure v-infinity
- capture cost included only for bodies with configured capture model

Validations:
- launch C3 11.081 km^2/s^2
- arrival v_inf 2.778 km/s

### direct_pareto_08

- id: `route_a44f44159e470f`
- engine: `porkchop.optimize_window`
- sequence: `EARTH -> MARS BARYCENTER`
- total cost: `3710.046 m/s`
- time of flight: `327.692 d`
- C3: `10.929656148559136`
- arrival v-inf: `2.8545075778929263`
- fidelity: `patched_conic_ephemeris`
- feasible: `True`
- Pareto: `False`
- certificate: `a44f44159e470fdc57ae82265432ff878925c5ecf91899e92746d6231aa87b59`

Assumptions:
- heliocentric Lambert arc on real ephemerides
- departure parking-orbit cost used when configured; otherwise departure v-infinity
- capture cost included only for bodies with configured capture model

Validations:
- launch C3 10.930 km^2/s^2
- arrival v_inf 2.855 km/s

### direct_pareto_06

- id: `route_55443dabbedea8`
- engine: `porkchop.optimize_window`
- sequence: `EARTH -> MARS BARYCENTER`
- total cost: `3746.274 m/s`
- time of flight: `258.462 d`
- C3: `11.76378916626027`
- arrival v-inf: `3.105190407245941`
- fidelity: `patched_conic_ephemeris`
- feasible: `True`
- Pareto: `True`
- certificate: `55443dabbedea8ed2fda3056f4ba9b13411c9f9db5aeb1df7805f25d9463cb6d`

Assumptions:
- heliocentric Lambert arc on real ephemerides
- departure parking-orbit cost used when configured; otherwise departure v-infinity
- capture cost included only for bodies with configured capture model

Validations:
- launch C3 11.764 km^2/s^2
- arrival v_inf 3.105 km/s

### direct_grid_best

- id: `route_3aba93786458ab`
- engine: `porkchop.optimize_window`
- sequence: `EARTH -> MARS BARYCENTER`
- total cost: `3687.603 m/s`
- time of flight: `350.769 d`
- C3: `10.414212417260643`
- arrival v-inf: `3.3288944518528236`
- fidelity: `patched_conic_ephemeris`
- feasible: `True`
- Pareto: `False`
- certificate: `3aba93786458ab0c80ddbb8a44396ced711c795dadb18709fcb0b1056c45ab9d`

Assumptions:
- heliocentric Lambert arc on real ephemerides
- departure parking-orbit cost used when configured; otherwise departure v-infinity
- capture cost included only for bodies with configured capture model

Validations:
- launch C3 10.414 km^2/s^2
- arrival v_inf 3.329 km/s

### direct_pareto_09

- id: `route_b4623ccb507a80`
- engine: `porkchop.optimize_window`
- sequence: `EARTH -> MARS BARYCENTER`
- total cost: `3687.603 m/s`
- time of flight: `350.769 d`
- C3: `10.414212417260643`
- arrival v-inf: `3.3288944518528236`
- fidelity: `patched_conic_ephemeris`
- feasible: `True`
- Pareto: `False`
- certificate: `b4623ccb507a80a9eed2d16d95b718b7c298d26c7322b3b5034846fc5eb822a7`

Assumptions:
- heliocentric Lambert arc on real ephemerides
- departure parking-orbit cost used when configured; otherwise departure v-infinity
- capture cost included only for bodies with configured capture model

Validations:
- launch C3 10.414 km^2/s^2
- arrival v_inf 3.329 km/s

### direct_pareto_05

- id: `route_0d2b2c5634c4c1`
- engine: `porkchop.optimize_window`
- sequence: `EARTH -> MARS BARYCENTER`
- total cost: `3780.980 m/s`
- time of flight: `235.385 d`
- C3: `12.565346732086883`
- arrival v-inf: `3.8937110326867606`
- fidelity: `patched_conic_ephemeris`
- feasible: `True`
- Pareto: `True`
- certificate: `0d2b2c5634c4c1d488776f562920829a4d0437ee090895e11eb67e5803924d1f`

Assumptions:
- heliocentric Lambert arc on real ephemerides
- departure parking-orbit cost used when configured; otherwise departure v-infinity
- capture cost included only for bodies with configured capture model

Validations:
- launch C3 12.565 km^2/s^2
- arrival v_inf 3.894 km/s

### direct_pareto_04

- id: `route_ea9b5405819713`
- engine: `porkchop.optimize_window`
- sequence: `EARTH -> MARS BARYCENTER`
- total cost: `3830.301 m/s`
- time of flight: `212.308 d`
- C3: `13.70859582006064`
- arrival v-inf: `5.036099525697785`
- fidelity: `patched_conic_ephemeris`
- feasible: `True`
- Pareto: `True`
- certificate: `ea9b540581971395826db8aa9d444696361d63c390b909d3c9e58bab29672315`

Assumptions:
- heliocentric Lambert arc on real ephemerides
- departure parking-orbit cost used when configured; otherwise departure v-infinity
- capture cost included only for bodies with configured capture model

Validations:
- launch C3 13.709 km^2/s^2
- arrival v_inf 5.036 km/s

### direct_pareto_knee

- id: `route_3a1673f6b2eac3`
- engine: `porkchop.optimize_window`
- sequence: `EARTH -> MARS BARYCENTER`
- total cost: `3945.413 m/s`
- time of flight: `189.231 d`
- C3: `16.39580705813856`
- arrival v-inf: `6.55974030932064`
- fidelity: `patched_conic_ephemeris`
- feasible: `True`
- Pareto: `True`
- certificate: `3a1673f6b2eac363661d1eb967932a3447918b43e2b0fdb1714ccb87d3766fd6`

Assumptions:
- heliocentric Lambert arc on real ephemerides
- departure parking-orbit cost used when configured; otherwise departure v-infinity
- capture cost included only for bodies with configured capture model

Validations:
- launch C3 16.396 km^2/s^2
- arrival v_inf 6.560 km/s

### direct_pareto_02

- id: `route_da892619bea14d`
- engine: `porkchop.optimize_window`
- sequence: `EARTH -> MARS BARYCENTER`
- total cost: `4218.570 m/s`
- time of flight: `166.154 d`
- C3: `22.878520488826293`
- arrival v-inf: `8.560759892883523`
- fidelity: `patched_conic_ephemeris`
- feasible: `True`
- Pareto: `True`
- certificate: `da892619bea14d083586ab02c3f6b68dfe7f869d5e59771c75e13ea52d4805d3`

Assumptions:
- heliocentric Lambert arc on real ephemerides
- departure parking-orbit cost used when configured; otherwise departure v-infinity
- capture cost included only for bodies with configured capture model

Validations:
- launch C3 22.879 km^2/s^2
- arrival v_inf 8.561 km/s

### direct_pareto_01

- id: `route_4f2e5702363048`
- engine: `porkchop.optimize_window`
- sequence: `EARTH -> MARS BARYCENTER`
- total cost: `4831.585 m/s`
- time of flight: `143.077 d`
- C3: `37.97012135259262`
- arrival v-inf: `11.222153060509065`
- fidelity: `patched_conic_ephemeris`
- feasible: `True`
- Pareto: `True`
- certificate: `4f2e5702363048a4c5b2c8e5dfdec88e7a59884cf9957f2cd143a2c6b23a1ef8`

Assumptions:
- heliocentric Lambert arc on real ephemerides
- departure parking-orbit cost used when configured; otherwise departure v-infinity
- capture cost included only for bodies with configured capture model

Validations:
- launch C3 37.970 km^2/s^2
- arrival v_inf 11.222 km/s

### direct_pareto_00

- id: `route_12a8b5753f0a5a`
- engine: `porkchop.optimize_window`
- sequence: `EARTH -> MARS BARYCENTER`
- total cost: `5214.304 m/s`
- time of flight: `120.000 d`
- C3: `47.77325517809634`
- arrival v-inf: `11.141019406154285`
- fidelity: `patched_conic_ephemeris`
- feasible: `True`
- Pareto: `True`
- certificate: `12a8b5753f0a5acbe4f7a6124184fc00200aa8d9212b35715e327fa264e0a0a9`

Assumptions:
- heliocentric Lambert arc on real ephemerides
- departure parking-orbit cost used when configured; otherwise departure v-infinity
- capture cost included only for bodies with configured capture model

Validations:
- launch C3 47.773 km^2/s^2
- arrival v_inf 11.141 km/s
