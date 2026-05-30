---
name: Bug report
about: Report a reproducible problem with Ariadne
title: "[bug] "
labels: bug
---

## What I expected to happen

(One sentence.)

## What actually happened

(One sentence + traceback / wrong number / etc.)

## Minimal reproducer

```python
import ariadne
# ...the 5-10 lines that show the problem
```

## Environment

- OS:
- Python version: (output of `python --version`)
- Ariadne version: (output of `python -c "import ariadne; print(ariadne.__version__)"`)
- Other relevant package versions: (numpy, scipy, spiceypy)

## Cross-check (optional but ideal)

If you can compare to GMAT / poliastro / a published value, please include that. The
honesty firewall says we'd rather find out we're wrong than ship a plausible mistake.
