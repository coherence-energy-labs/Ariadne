# Ariadne Engine Improvement Benchmark

This report compares engine settings on identical held-out cases.
It does not include closure gates.

- corpus: `data\benchmarks\real_corpus_mpc_500\labelled_cases.jsonl`
- total cases: `500`
- train cases: `363`
- eval cases: `137`
- adversarial eval cases: `822`

## Held-Out Real Corpus

| metric | baseline engine | tuned engine | delta |
|---|---:|---:|---:|
| accuracy | 1.000000 | 1.000000 | 0.000000 |
| safe_accuracy | 1.000000 | 1.000000 | 0.000000 |
| macro_precision | 1.000000 | 1.000000 | 0.000000 |
| macro_recall | 1.000000 | 1.000000 | 0.000000 |
| macro_f1 | 1.000000 | 1.000000 | 0.000000 |
| nll | 0.002869 | 0.000046 | -0.002823 |
| brier | 0.000076 | 3.102e-08 | -0.000076 |
| ece | 0.002830 | 0.000046 | -0.002784 |
| failures | 0 | 0 | 0 |

## Held-Out Adversarial Mutations

| metric | baseline engine | tuned engine | delta |
|---|---:|---:|---:|
| accuracy | 1.000000 | 1.000000 | 0.000000 |
| safe_accuracy | 1.000000 | 1.000000 | 0.000000 |
| macro_precision | 1.000000 | 1.000000 | 0.000000 |
| macro_recall | 1.000000 | 1.000000 | 0.000000 |
| macro_f1 | 1.000000 | 1.000000 | 0.000000 |
| nll | 0.014647 | 0.005693 | -0.008954 |
| brier | 0.003375 | 0.001709 | -0.001666 |
| ece | 0.012029 | 0.004515 | -0.007513 |
| failures | 0 | 0 | 0 |

## Interpretation

- Accuracy deltas measure class decisions only.
- Safe accuracy gives credit for correct abstain/follow-up behavior.
- Lower NLL, Brier, and ECE mean better calibrated confidence.
- The tuned engine is trained only on the deterministic train split and scored on held-out eval cases.
