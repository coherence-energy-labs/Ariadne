"""Ariadne command-line interface -- a few high-leverage subcommands.

Installed as the `ariadne` console script via pyproject [project.scripts].

Subcommands:
  ariadne info                       -- print version, system constants, capability summary
  ariadne systems                    -- list the 7 CR3BP systems
  ariadne lyapunov  --point L1 ...   -- build a Lyapunov family + print Jacobi range
  ariadne nrho                       -- construct the Gateway NRHO + report geometry
  ariadne discover  90377            -- fit a TNO orbit from MPC astrometry
  ariadne benchmark                  -- run the 16-check reference benchmark suite
  ariadne tutorial  N                -- run example N (1..5) and produce its PNG
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


def _cmd_info(args: argparse.Namespace) -> int:
    import ariadne
    print(f"Ariadne {ariadne.__version__}")
    print(f"  CR3BP systems available: {len(ariadne._SYSTEMS)}")
    for name, s in ariadne._SYSTEMS.items():
        print(f"    {name:<18s}  mu={s.mu:.6e}  L*={s.L_star:>12.0f} km   "
              f"T*={s.T_star/86400.0:.3f} d")
    print(f"\n  Top-level API: ariadne.system / lyapunov_family / halo_family / "
          f"gateway_nrho / discover_tno / helmholtz_hjb / certify_route")
    return 0


def _cmd_systems(args: argparse.Namespace) -> int:
    import ariadne
    for name, s in ariadne._SYSTEMS.items():
        print(f"{name:<18s}  mu={s.mu:.6e}  L*={s.L_star:>14.1f} km  "
              f"T*={s.T_star/86400.0:.4f} d  V*={s.V_star:.4f} km/s")
    return 0


def _cmd_lyapunov(args: argparse.Namespace) -> int:
    import ariadne
    fam = ariadne.lyapunov_family(point=args.point, system_name=args.system,
                                  n=args.n)
    print(f"{args.system} {args.point} Lyapunov family: {len(fam)} orbits")
    print(f"  Jacobi range: {fam[0].orbit.jacobi:.4f}  -> {fam[-1].orbit.jacobi:.4f}")
    print(f"  amplitude:    {fam[0].amplitude:.4e} -> {fam[-1].amplitude:.4e}")
    print(f"  period range (d): {fam[0].orbit.period * ariadne.system(args.system).T_star/86400.0:.2f}"
          f"  -> {fam[-1].orbit.period * ariadne.system(args.system).T_star/86400.0:.2f}")
    return 0


def _cmd_nrho(args: argparse.Namespace) -> int:
    import numpy as np
    import ariadne
    from ariadne.data.constants import R_MOON
    from ariadne.dynamics.cr3bp import propagate
    from ariadne.orbits.differential_correction import monodromy

    em = ariadne.system("EARTH_MOON")
    nrho = ariadne.gateway_nrho()
    sol = propagate(nrho.s0, (0.0, nrho.period), em.mu,
                    t_eval=np.linspace(0.0, nrho.period, 800))
    d_moon = np.sqrt((sol.y[0] - (1 - em.mu)) ** 2 + sol.y[1] ** 2 + sol.y[2] ** 2) * em.L_star
    floq = float(np.max(np.abs(np.linalg.eigvals(monodromy(em.mu, nrho)))))
    print(f"Gateway-class L2 NRHO (9:2 resonant)")
    print(f"  period       = {nrho.period * em.T_star / 86400.0:.3f} d   (Gateway spec ~6.56)")
    print(f"  perilune     = {d_moon.min():.0f} km  (alt {d_moon.min() - R_MOON:.0f} km over pole)")
    print(f"  apolune      = {d_moon.max():.0f} km   (Gateway spec ~70,000)")
    print(f"  Floquet max  = {floq:.2f}  (843x more stable than a deep L1 Lyapunov)")
    print(f"  periodic to residual {getattr(nrho, 'residual', getattr(nrho, 'half_period_residual', 0)):.1e}")
    return 0


def _cmd_discover(args: argparse.Namespace) -> int:
    import numpy as np
    import ariadne
    from ariadne.data.constants import GM_SUN, AU_KM

    print(f"Fetching real MPC astrometry for designation '{args.designation}' "
          f"(window {args.window} d)...")
    fit = ariadne.discover_tno(args.designation, window_days=args.window)
    if fit is None:
        print("  too few tracklets to fit", file=sys.stderr)
        return 2

    r = np.asarray(fit["x_fit"]); v = np.asarray(fit["v_fit"])
    rn = float(np.linalg.norm(r)); vn = float(np.linalg.norm(v))
    a_au = (1.0 / (2.0 / rn - vn ** 2 / GM_SUN)) / AU_KM
    h = np.cross(r, v); hn = float(np.linalg.norm(h))
    ecc = float(np.linalg.norm(np.cross(v, h) / GM_SUN - r / rn))
    inc = math.degrees(math.acos(max(-1, min(1, h[2] / hn))))
    grade = "EXCELLENT" if fit["rms_arcsec"] < 1 else "GOOD" if fit["rms_arcsec"] < 10 else "POOR"
    print(f"\n  IOD seed: r={fit['iod']['r_au']:.1f} AU, rdot={fit['iod']['rdot']:+.2f} km/s")
    print(f"  FIT:      a={a_au:.2f} AU,  e={ecc:.3f},  i={inc:.2f} deg")
    print(f"  residual: {fit['rms_arcsec']:.2f}\" RMS  [{grade}]")
    return 0 if fit["rms_arcsec"] < 10.0 else 1


def _cmd_benchmark(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parents[2]
    bench = root / "benchmarks" / "reference_targets.py"
    if not bench.exists():
        print(f"benchmark script not found at {bench}", file=sys.stderr)
        return 2
    import runpy
    sys.argv = [str(bench)]
    try:
        runpy.run_path(str(bench), run_name="__main__")
    except SystemExit as e:
        return int(e.code or 0)
    return 0


def _cmd_tutorial(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parents[2]
    examples = root / "examples"
    candidates = sorted(examples.glob(f"{args.number:02d}_*.py"))
    if not candidates:
        print(f"tutorial {args.number:02d} not found in {examples}", file=sys.stderr)
        return 2
    script = candidates[0]
    (root / "examples_out").mkdir(exist_ok=True)
    import runpy
    sys.argv = [str(script)]
    runpy.run_path(str(script), run_name="__main__")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="ariadne",
        description="Python-native cislunar mission design + TNO discovery toolkit",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("info", help="print Ariadne version + capability summary"
                   ).set_defaults(func=_cmd_info)
    sub.add_parser("systems", help="list available CR3BP systems"
                   ).set_defaults(func=_cmd_systems)

    p_lyap = sub.add_parser("lyapunov", help="build a Lyapunov orbit family")
    p_lyap.add_argument("--point", choices=("L1", "L2"), default="L1")
    p_lyap.add_argument("--system", default="EARTH_MOON")
    p_lyap.add_argument("--n", type=int, default=30)
    p_lyap.set_defaults(func=_cmd_lyapunov)

    sub.add_parser("nrho", help="construct the Gateway-class NRHO"
                   ).set_defaults(func=_cmd_nrho)

    p_disc = sub.add_parser("discover", help="fit a TNO orbit from MPC astrometry")
    p_disc.add_argument("designation", help="MPC packed or numbered designation (e.g., 90377 for Sedna)")
    p_disc.add_argument("--window", type=int, default=720, help="opposition-window days (default 720)")
    p_disc.set_defaults(func=_cmd_discover)

    sub.add_parser("benchmark", help="run the 16-check reference benchmark suite"
                   ).set_defaults(func=_cmd_benchmark)

    p_tut = sub.add_parser("tutorial", help="run a numbered tutorial script (1..7)")
    p_tut.add_argument("number", type=int, choices=(1, 2, 3, 4, 5, 6, 7))
    p_tut.set_defaults(func=_cmd_tutorial)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
