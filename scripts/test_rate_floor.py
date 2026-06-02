"""Rate floor for single-snapshot velocity from PSF elongation.

Controlled test: inject synthetic trailed point sources at a GRID of known
angular rates into a real DECam CCD (real sky + noise), measure each one's
second-moment elongation, convert to an implied rate, and find the rate at
which the elongation signal separates from the round (rate=0) baseline.

This answers the real question the first test couldn't (it had no rate
dynamic range -- all known asteroids were main-belt at ~35"/hr): at what
angular rate does a single 90 s exposure yield a usable velocity?
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

FITS = "data/real_decam_recovery/c4d_240824_013234_ooi_r_v1.fits.fz"
PIXSCALE = 0.263
TEXP = 90.0
STAMP = 18


def inject_trail(img, x, y, rate_arcsec_hr, pa_deg, total_flux, fwhm_px=3.8):
    """Add a PSF-convolved linear trail (rate over TEXP) centered at x,y."""
    L_arcsec = rate_arcsec_hr * (TEXP / 3600.0)
    L_px = L_arcsec / PIXSCALE
    sigma = fwhm_px / 2.3548
    n = max(int(L_px * 3), 1)
    dx = 0.5 * L_px * math.sin(math.radians(pa_deg))
    dy = 0.5 * L_px * math.cos(math.radians(pa_deg))
    fpp = total_flux / n
    half = int(math.ceil(4 * sigma + L_px))
    H, W = img.shape
    for t in np.linspace(-1, 1, n):
        cx = x + t * dx; cy = y + t * dy
        x0 = max(0, int(cx) - half); x1 = min(W, int(cx) + half + 1)
        y0 = max(0, int(cy) - half); y1 = min(H, int(cy) + half + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        img[y0:y1, x0:x1] += (fpp / (2 * math.pi * sigma ** 2)) * np.exp(
            -((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))


def measure(stamp):
    from ariadne.discovery.imaging.morphology import _second_moments, _theta_from_moments
    bg = float(np.median(stamp))
    sub = np.clip(stamp - bg, 0, None)
    if sub.sum() <= 0:
        return None
    yy, xx = np.mgrid[:stamp.shape[0], :stamp.shape[1]]
    xc = float((sub * xx).sum() / sub.sum()); yc = float((sub * yy).sum() / sub.sum())
    a, b = _second_moments(stamp, xc, yc, bg)
    th = _theta_from_moments(stamp, xc, yc, bg)
    return a, b, th


def main():
    from ariadne.discovery.imaging.decam_instcal import load_decam_instcal
    inst = load_decam_instcal(FITS, read_dqm=False)
    ccd = next(c for c in inst.ccds if c.wcs is not None and c.magzero > 0)
    base = np.asarray(ccd.science, float)
    H, W = base.shape
    zp = ccd.magzero
    rng = np.random.default_rng(0)
    # blank-ish patches: pick random positions (sky-dominated)
    flux_for_mag = lambda m: 10 ** ((zp - m) / 2.5)

    print(f"CCD {ccd.name}, exptime={TEXP}s, pixscale={PIXSCALE}\"/px, ZP={zp:.1f}")
    print(f"{'rate(\"/hr)':>10s} {'trail(px)':>9s} {'a/b':>6s} "
          f"{'a^2-b^2':>8s} {'impl_rate':>10s} {'dPA':>5s}  (mag=18, n=40)")
    print("-" * 64)
    # baseline a^2-b^2 from rate=0 injections (round PSF on real noise)
    rates = [0, 10, 30, 60, 120, 250, 500, 1000, 2000]
    summary = []
    for rate in rates:
        ab2 = []; impl = []; dpa = []
        true_pa = 35.0
        for _ in range(40):
            x = rng.uniform(200, W - 200); y = rng.uniform(200, H - 200)
            work = base[int(y) - STAMP - 30:int(y) + STAMP + 30,
                         int(x) - STAMP - 30:int(x) + STAMP + 30].copy()
            cx, cy = work.shape[1] // 2, work.shape[0] // 2
            inject_trail(work, cx, cy, rate, true_pa, flux_for_mag(18.0))
            st = work[cy - STAMP:cy + STAMP + 1, cx - STAMP:cx + STAMP + 1]
            m = measure(st)
            if m is None:
                continue
            a, b, th = m
            ab2.append(a ** 2 - b ** 2)
            impl.append(a ** 2 - b ** 2)
            dpa.append(abs((math.degrees(th) - true_pa + 90) % 180 - 90))
        ab2 = np.array(ab2)
        L_arcsec = rate * (TEXP / 3600.0); L_px = L_arcsec / PIXSCALE
        # implied rate from median excess over rate=0 baseline
        excess = np.median(ab2) - summary[0][1] if summary else 0.0
        impl_rate = (math.sqrt(12 * excess) if excess > 0 else 0.0) * PIXSCALE / TEXP * 3600
        summary.append((rate, float(np.median(ab2)), float(np.std(ab2)),
                          impl_rate, float(np.median(dpa))))
        # a/b
        print(f"{rate:10d} {L_px:9.1f} {'':6s} {np.median(ab2):8.2f} "
              f"{impl_rate:9.0f}\" {np.median(dpa):5.0f}")

    # Find the floor: rate where median(a^2-b^2) exceeds baseline + 3 sigma
    base_med = summary[0][1]; base_sig = summary[0][2]
    print(f"\n  rate=0 baseline a^2-b^2 = {base_med:.2f} +/- {base_sig:.2f} px^2")
    floor = None
    for rate, med, sig, ir, dp in summary[1:]:
        if med > base_med + 3 * base_sig:
            floor = rate; break
    print(f"  3-sigma detection floor: rate >= {floor}\"/hr "
          f"(main belt ~35\"/hr, NEOs 100s-1000s\"/hr)")
    print(f"  => single-snapshot velocity is recoverable for "
          f"{'FAST movers/NEOs' if floor and floor > 60 else 'most movers'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
