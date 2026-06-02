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
| nll | 0.002869 | 1.590e-11 | -0.002869 |
| brier | 0.000076 | 5.666e-21 | -0.000076 |
| ece | 0.002830 | 1.590e-11 | -0.002830 |
| failures | 0 | 0 | 0 |

## Held-Out Adversarial Mutations

| metric | baseline engine | tuned engine | delta |
|---|---:|---:|---:|
| accuracy | 0.810219 | 0.810219 | 0.000000 |
| safe_accuracy | 1.000000 | 1.000000 | 0.000000 |
| macro_precision | 1.000000 | 1.000000 | 0.000000 |
| macro_recall | 0.738715 | 0.738715 | 0.000000 |
| macro_f1 | 0.847685 | 0.847685 | 0.000000 |
| nll | 0.091268 | 0.123354 | 0.032086 |
| brier | 0.026567 | 0.020404 | -0.006163 |
| ece | 0.038319 | 0.015420 | -0.022899 |
| failures | 156 | 156 | 0 |

## Interpretation

- Accuracy deltas measure class decisions only.
- Safe accuracy gives credit for correct abstain/follow-up behavior.
- Lower NLL, Brier, and ECE mean better calibrated confidence.
- The tuned engine is trained only on the deterministic train split and scored on held-out eval cases.
