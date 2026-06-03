"""Cockpit job worker: run one (possibly slow) Ariadne engine call and write the
result back into the job's JSON file. Spawned by the dashboard so the UI never
blocks on the 6-40s mission-design computations.

  python scripts/cockpit_worker.py <jobfile.json>

The jobfile starts as {"id","engine","params"} and is updated in place with
status (queued->running->done/error), result, and timing.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _periods_days(fam, s):
    return [round(float(getattr(o, "period", 0.0)) * s.T_star / 86400.0, 3) for o in fam]


CAPS = {
    1: ("Lyapunov family", "Earth-Moon L1 planar Lyapunov orbit family", "01_lyapunov_family.png"),
    2: ("Gateway NRHO", "9:2 near-rectilinear halo orbit + monodromy stability", "02_gateway_nrho.png"),
    3: ("Cislunar transport", "Manifold transport graph: the Earth-Moon highways", "03_manifold_transport.png"),
    4: ("TNO orbit fit", "Fit a real TNO orbit from MPC astrometry", "04_tno_orbit_fit.png"),
    5: ("Coherence-HJB", "6D Hamilton-Jacobi-Bellman optimal-control value function", "05_helmholtz_hjb.png"),
    6: ("VEEGA to Jupiter", "Venus-Earth-Earth gravity-assist trajectory", "06_veega_jupiter.png"),
    7: ("Lambert porkchop", "Earth-Mars launch-energy porkchop", "07_lambert_porkchop.png"),
    8: ("ZTF alert filter", "ZTF alert stream -> moving-object candidate pipeline", None),
    9: ("DECam discovery", "Image-based discovery pipeline (synthetic end-to-end)", None),
    10: ("Nightly operations", "End-to-end nightly discovery operations loop", None),
}


def run(engine, p):
    if engine == "capability":
        import glob
        import os
        import subprocess
        n = int(p["n"])
        scripts = sorted(glob.glob(str(ROOT / "examples" / f"{n:02d}_*.py")))
        if not scripts:
            raise ValueError(f"no example tutorial {n}")
        env = dict(os.environ, MPLBACKEND="Agg")          # headless figures
        r = subprocess.run([sys.executable, scripts[0]], cwd=str(ROOT), env=env,
                           capture_output=True, text=True, timeout=900)
        title, desc, fig = CAPS.get(n, (f"tutorial {n}", "", None))
        out = (r.stdout or "").strip().splitlines()
        tail = "\n".join(out[-14:]) or (r.stderr or "")[-500:]
        # the engines run fine; only matplotlib figure-save crashes on Python 3.14
        # (Patch.__deepcopy__ C-stack overflow). Report engine results + be honest.
        note = None
        err = r.stderr or ""
        if r.returncode != 0 and ("RecursionError" in err or "Stack overflow" in err):
            note = ("engine ran (results above); live figure save hit the matplotlib "
                    "Patch deepcopy bug on Python 3.14 - the gallery shows the pre-rendered figure")
        res = {"capability": title, "script": Path(scripts[0]).name, "exit_code": r.returncode,
               "figure": fig, "output_tail": tail}
        if note:
            res["note"] = note
        return res

    import ariadne
    s = ariadne.EARTH_MOON
    if engine == "nrho":
        o = ariadne.gateway_nrho()
        return {"period_days": round(o.period * s.T_star / 86400.0, 3),
                "jacobi_constant": round(float(o.jacobi), 5),
                "z_amplitude_ndim": round(float(o.z_amplitude), 5),
                "point": str(o.point), "residual": float(getattr(o, "residual", 0.0))}
    if engine in ("lyapunov", "halo"):
        point = p.get("point", "L1"); n = max(2, min(int(p.get("n", 10)), 60))
        fam = (ariadne.lyapunov_family if engine == "lyapunov" else ariadne.halo_family)(point, n=n)
        per = _periods_days(fam, s)
        return {"family": engine, "point": point, "n_orbits": len(fam),
                "period_days_min": min(per) if per else None,
                "period_days_max": max(per) if per else None,
                "jacobi_first": round(float(getattr(fam[0], "jacobi", 0.0)), 5) if fam else None}
    if engine == "discover_tno":
        fit = ariadne.discover_tno(str(p["designation"]), window_days=int(p.get("window_days", 720)))
        if not fit:
            return {"note": "fewer than 4 tracklets in the densest opposition window"}
        out = {}
        src = fit if isinstance(fit, dict) else {k: getattr(fit, k) for k in dir(fit) if not k.startswith("_")}
        for k, v in src.items():
            if isinstance(v, (int, float, str, bool)):
                out[k] = round(v, 4) if isinstance(v, float) else v
        return out or {"note": "fit returned, no scalar fields"}
    if engine == "system":
        sysobj = ariadne.system(p.get("name", "EARTH_MOON"))
        return {"name": sysobj.name, "mu": round(float(sysobj.mu), 8),
                "L_star_km": round(float(sysobj.L_star), 1),
                "T_star_days": round(float(sysobj.T_star) / 86400.0, 4),
                "V_star_kms": round(float(sysobj.V_star), 5),
                "primary": str(sysobj.primary), "secondary": str(sysobj.secondary)}
    raise ValueError(f"unknown engine: {engine}")


def main():
    jf = Path(sys.argv[1])
    job = json.loads(jf.read_text())
    job["status"] = "running"; job["started"] = time.time()
    jf.write_text(json.dumps(job))
    try:
        job["result"] = run(job["engine"], job.get("params", {}))
        job["status"] = "done"
    except Exception as e:
        job["status"] = "error"; job["error"] = f"{type(e).__name__}: {e}"
    job["finished"] = time.time()
    jf.write_text(json.dumps(job))


if __name__ == "__main__":
    main()
