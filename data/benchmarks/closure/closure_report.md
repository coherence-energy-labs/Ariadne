# Ariadne Closure Report

- status: complete
- readiness_score: 1.000000
- critical_failures: 0
- warnings: 0
- blocking_residuals: 0
- certificate_hash: `4a65a794e70cdf3d42025890a1257ee2e28b1d9b96a71edf434f3361ee52375e`

## Gate Results

| gate | subsystem | status | severity | message |
|---|---|---:|---:|---|
| `dream_calibration:dream_run:required` | dream_calibration | pass | critical |  |
| `dream_calibration:experiment_count:>=` | dream_calibration | pass | critical |  |
| `discovery_inference:labelled_benchmark:required` | discovery_inference | pass | critical |  |
| `discovery_inference:accuracy:>=` | discovery_inference | pass | critical |  |
| `discovery_inference:safe_accuracy:>=` | discovery_inference | pass | critical |  |
| `discovery_inference:macro_f1:>=` | discovery_inference | pass | critical |  |
| `discovery_inference:ece:<=` | discovery_inference | pass | critical |  |
| `discovery_robustness:adversarial_benchmark:required` | discovery_robustness | pass | critical |  |
| `discovery_robustness:safe_accuracy:>=` | discovery_robustness | pass | critical |  |
| `solar_navigator:navigator_benchmark:required` | solar_navigator | pass | critical |  |
| `visuals:figure_audit:required` | visuals | pass | critical |  |
| `visuals:reviewer_visual_contract:required` | visuals | pass | critical |  |
| `visuals:png_valid_fraction:>=` | visuals | pass | critical |  |
| `visuals:n_png:>=` | visuals | pass | critical |  |
| `visuals:semantic_route_fraction:>=` | visuals | pass | critical |  |
| `trajectory_certification:route_certificate:soft_required` | trajectory_certification | pass | major |  |
| `transport_search:admissibility_benchmark:required` | transport_search | pass | critical |  |
| `transport_search:speedup_astar_vs_brute:>=` | transport_search | pass | critical |  |
| `artifact_integrity:artifact_manifest:required` | artifact_integrity | pass | critical |  |
| `artifact_integrity:file_count:>=` | artifact_integrity | pass | critical |  |

## Highest-Priority Residuals

| residual | subsystem | category | priority | action |
|---|---|---|---:|---|
| `route_multifidelity_promotion_ladder` | trajectory_certification | flight_grade_proof | 0.462 | Extend strict promotion to moon-system ephemerides, low-thrust/DSM arcs, and installed GMAT/Monte external replays. |
| `navigator_non_earth_origin_extension` | solar_navigator | model_scope | 0.416 | Add non-Earth gravity-assist templates and benchmark coverage for non-Earth departure bodies. |
| `deep_field_rate_constrained_cross_night_linking` | discovery_pipeline | operational_scaling | 0.352 | Benchmark the rate-constrained linker on near-ecliptic deep cadence data and track runtime/false-link curves. |
| `discovery_long_arc_nbody_default` | discovery_inference | fidelity_default | 0.324 | Expand real-corpus regression coverage for long-arc N-body fits and monitor fallback rates. |
| `reviewer_grade_visual_contract` | visuals | explainability | 0.289 | Add optional OCR/metadata sidecar scoring for units, labels, uncertainty overlays, and scale bars. |
