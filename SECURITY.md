# Security policy

Ariadne is a scientific/engineering library — it doesn't handle credentials, network
auth, or untrusted input by default. The threat surface is small. Still:

## Supported versions

The latest release on PyPI receives security fixes. Older versions don't.

| Version | Supported          |
|---------|--------------------|
| ≥1.0.0  | yes                |
| < 1.0.0 | no                 |

## Reporting a vulnerability

If you find a real security issue (not a bug — a *security* issue: arbitrary code
execution, credential leak, supply-chain attack), please email the maintainers privately
rather than opening a public GitHub issue.

We will:

1. Acknowledge receipt within a few days.
2. Investigate the severity and reproduce internally.
3. Coordinate a fix and disclosure timeline with you.
4. Credit you in the release notes unless you prefer not.

## What counts as a security issue

- Arbitrary code execution from untrusted file input (e.g., an attacker-crafted SPICE
  kernel, MPC astrometry file, atlas HDF5).
- Path traversal / write-anywhere when loading user-supplied paths.
- Network requests to attacker-controlled URLs.
- Credential or token leaks (we don't store any, but if we did).
- Dependency vulnerabilities that affect Ariadne specifically.

## What doesn't count (please open a normal issue instead)

- Numerical bugs (orbit fits, transfer Δv, manifold cuts off by some factor) — those
  go to the regular bug tracker. The honesty firewall says we want every numerical bug
  reported publicly with a reproducer.
- Performance issues.
- Misleading documentation.
- API ergonomics complaints.
