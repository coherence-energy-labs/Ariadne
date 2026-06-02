# Ariadne Improvement Benchmark Report

| metric | before | after | delta |
|---|---:|---:|---:|
| labelled inference accuracy | 0.928571 | 1 | 0.0714286 |
| labelled inference safe_accuracy | n/a | 1 | n/a |
| labelled inference macro_f1 | 0.69697 | 1 | 0.30303 |
| labelled inference macro_recall | 0.727273 | 1 | 0.272727 |
| labelled inference ECE | 0.290597 | 7.04514e-12 | -0.290597 |
| closure readiness_score | n/a | 1 | n/a |
| adversarial safe_accuracy | n/a | 0.946667 | n/a |
| transport A* speedup vs brute | n/a | 3x | n/a |

## Notes
- Baseline real_run.json is an older built-in labelled benchmark artifact, not the same sample as the MPC-500 corpus.
- MPC-500 is real MPCORB-derived labelled data; adversarial suite is proxy/synthetic mutation stress testing.
- Before/after deltas are meaningful for headline inference metrics but should not be read as paired-sample statistics.
