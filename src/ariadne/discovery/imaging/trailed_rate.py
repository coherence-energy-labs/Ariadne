"""Single-snapshot angular RATE from PSF trailing.

A moving object dragged at angular rate w over exposure time T_exp leaves a
trail of length L = w*T_exp, which convolves with the seeing PSF to add
variance L^2/12 ALONG the motion. The perpendicular axis stays at the PSF
width, so the source's second-moment tensor encodes the velocity vector of
a SINGLE exposure:

    L_px      = sqrt(12 * (lambda_major - lambda_minor - psf_anisotropy))
    rate      = L_px * pixscale / T_exp
    PA_motion = orientation of the major axis, rotated to sky via the WCS

Validated by injection into real DECam pixels: implied rate tracks truth
from ~30 to ~500"/hr with PA recovered to ~20 deg. Strongest for fast
movers / NEOs (which trail most) -- exactly the high-value targets. Below
~30"/hr the per-object trail (<3 px on ~1" seeing) approaches the noise.

This is what lets a single frame carry velocity, which (a) collapses the
inter-night linking search and (b) seeds a single-tracklet orbit.

Public API:
  measure_psf_tensor(stamp) -> (lambda_major, lambda_minor, theta_pix)
  stellar_psf_anisotropy(image, ...) -> float   (instrumental lambda_maj-min)
  rate_from_stamp(stamp, *, psf_aniso, pixscale, t_exp, wcs?, x?, y?) -> RateEstimate
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class RateEstimate:
    rate_arcsec_hr: float
    pa_deg_sky: float | None     # sky-frame PA (deg, mod 180); None if no WCS
    pa_deg_pix: float            # pixel-frame major-axis PA (deg, mod 180)
    trail_px: float
    snr: float
    lambda_major: float
    lambda_minor: float


def measure_psf_tensor(stamp: np.ndarray):
    """Flux-weighted second-moment eigenvalues (px^2) + major-axis PA (rad).

    Returns (lambda_major, lambda_minor, theta_pix). Background is the stamp
    median; flux weights are background-subtracted, clipped at 0. NaN pixels
    (real CCD chip gaps / bad-pixel masks) are set to the background so they
    contribute zero weight rather than poisoning the moments.
    """
    if not np.isfinite(stamp).any():
        return 0.0, 0.0, 0.0
    bg = float(np.nanmedian(stamp))
    stamp = np.where(np.isfinite(stamp), stamp, bg)
    sub = np.clip(stamp - bg, 0.0, None)
    tot = sub.sum()
    if tot <= 0:
        return 0.0, 0.0, 0.0
    yy, xx = np.mgrid[:stamp.shape[0], :stamp.shape[1]]
    xc = float((sub * xx).sum() / tot)
    yc = float((sub * yy).sum() / tot)
    dx = xx - xc; dy = yy - yc
    Mxx = float((sub * dx * dx).sum() / tot)
    Myy = float((sub * dy * dy).sum() / tot)
    Mxy = float((sub * dx * dy).sum() / tot)
    tr = Mxx + Myy
    diff = math.sqrt(max(0.0, (Mxx - Myy) ** 2 + 4 * Mxy ** 2))
    lam_maj = 0.5 * (tr + diff)
    lam_min = 0.5 * (tr - diff)
    theta = 0.5 * math.atan2(2.0 * Mxy, Mxx - Myy)   # major-axis PA, pixel frame
    return lam_maj, lam_min, theta


def _fit_star_sigma(stamp: np.ndarray, sig_guess: float):
    """Fit a 2-D circular Gaussian (+background) to a star stamp; return sigma
    (px) or None. Fitting the CORE + a background level is robust to sky noise,
    unlike clipped second moments over a large stamp (which inflate the width
    when noise dominates the wings -- the failure mode that mis-measured a 3px
    PSF as 13px)."""
    try:
        from scipy.optimize import curve_fit
    except ImportError:
        return None
    H, W = stamp.shape
    finite = np.isfinite(stamp)
    if finite.sum() < 0.5 * stamp.size:
        return None
    bg0 = float(np.nanmedian(stamp))
    z = np.where(finite, stamp, bg0).astype(float)
    amp0 = float(np.nanmax(z) - bg0)
    if amp0 <= 0:
        return None
    # reject SATURATED stars: a real PSF has a single sharp peak; a saturated
    # star (or CR cluster) has a flat top -- many pixels within ~1% of the peak.
    # (Scale-independent: works whether full-well is 6e4 ADU or a synthetic 600.)
    if int(np.sum(z >= bg0 + 0.99 * amp0)) > 5:
        return None
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0

    def model(_xdata, amp, x0, y0, sig, bg):
        return (bg + amp * np.exp(-(((xx - x0) ** 2 + (yy - y0) ** 2)
                                    / (2.0 * sig * sig)))).ravel()

    try:
        popt, _ = curve_fit(
            model, None, z.ravel(),
            p0=[amp0, cx, cy, max(sig_guess, 0.8), bg0],
            bounds=([0.0, cx - 3, cy - 3, 0.5, bg0 - abs(amp0) - 1],
                    [np.inf, cx + 3, cy + 3, 15.0, bg0 + abs(amp0) + 1]),
            maxfev=2000)
        return float(popt[3])
    except Exception:
        return None


def measure_image_fwhm(image: np.ndarray, *, fwhm_guess: float = 4.0,
                         n_stars: int = 25, min_snr: float = 50.0,
                         max_iter: int = 3, isolation_factor: float = 4.0):
    """Robustly measure the median stellar PSF FWHM (px) of an image -- the
    "what am I looking at / what px should I use" brain for detection.

    Selects bright, ISOLATED, unsaturated, non-edge stars and fits each with a
    2-D Gaussian on a stamp scaled to the current FWHM estimate, then iterates
    (the stamp size tracks the PSF, so neither a too-small nor a noise-dominated
    too-large stamp biases it). Returns the sigma-clipped median FWHM, or None
    if no usable stars (caller then falls back to a default). Automatic and
    self-calibrating on any field -- not a hardcoded width."""
    try:
        from photutils.detection import DAOStarFinder
        from astropy.stats import sigma_clipped_stats
    except ImportError:
        return None
    data = np.asarray(image, float)
    fin = np.isfinite(data)
    if fin.sum() < 500:
        return None
    _m, med, std = sigma_clipped_stats(data[fin], sigma=3.0)
    if not np.isfinite(std) or std <= 0:
        return None
    H, W = data.shape
    work = np.where(fin, data, med) - med
    fwhm = float(fwhm_guess)
    result = None
    for _ in range(max_iter):
        try:
            tbl = DAOStarFinder(fwhm=max(fwhm, 1.5), threshold=min_snr * std)(work)
        except Exception:
            break
        if tbl is None or len(tbl) == 0:
            break
        xc = "x_centroid" if "x_centroid" in tbl.colnames else "xcentroid"
        yc = "y_centroid" if "y_centroid" in tbl.colnames else "ycentroid"
        rows = sorted(list(tbl), key=lambda r: -float(r["flux"]))
        xs = np.array([float(r[xc]) for r in rows])
        ys = np.array([float(r[yc]) for r in rows])
        half = int(np.ceil(3.0 * fwhm))
        sig_guess = fwhm / 2.3548
        sigs = []
        for k in range(len(rows)):
            x, y = xs[k], ys[k]
            xi, yi = int(round(x)), int(round(y))
            if xi < half or yi < half or xi >= W - half or yi >= H - half:
                continue
            d2 = (xs - x) ** 2 + (ys - y) ** 2
            d2[k] = np.inf
            if d2.size > 1 and math.sqrt(float(d2.min())) < isolation_factor * fwhm:
                continue                                   # blended neighbour
            st = data[yi - half:yi + half + 1, xi - half:xi + half + 1]
            s = _fit_star_sigma(st, sig_guess)             # rejects saturated/flat-top
            if s is not None and 0.5 < s < 15.0:
                sigs.append(s)
            if len(sigs) >= n_stars:
                break
        if not sigs:
            break
        new_fwhm = float(np.median(sigs)) * 2.3548
        result = new_fwhm
        if abs(new_fwhm - fwhm) < 0.1 * fwhm:
            break
        fwhm = float(np.clip(new_fwhm, 1.5, 30.0))
    return result


def stellar_psf_fwhm(image: np.ndarray, *, fwhm_px: float = 4.0,
                       n_stars: int = 60, stamp: int = 18,
                       min_snr: float = 30.0) -> float:
    """Median PSF FWHM (px) measured from bright round stars -- the ACTUAL
    seeing in this image. A trailed-PSF fit MUST use this, not an assumed
    value: a fixed-PSF model mistakes seeing-width mismatch for trail and
    biases the rate high. Returns the supplied default if unmeasurable.

    Delegates to the robust Gaussian-fit estimator `measure_image_fwhm` first
    (accurate + saturation-rejecting); only falls back to the legacy moment
    method if that cannot measure, so every consumer auto-upgrades."""
    robust = measure_image_fwhm(image, fwhm_guess=fwhm_px)
    if robust is not None and np.isfinite(robust) and 1.0 < robust < 25.0:
        return float(robust)
    try:
        from photutils.detection import DAOStarFinder
        from astropy.stats import sigma_clipped_stats
    except ImportError:
        return fwhm_px
    data = np.asarray(image, float)
    _m, med, std = sigma_clipped_stats(data, sigma=3.0)
    tbl = DAOStarFinder(fwhm=fwhm_px, threshold=min_snr * std)(data - med)
    if tbl is None or len(tbl) == 0:
        return fwhm_px
    xcol = "x_centroid" if "x_centroid" in tbl.colnames else "xcentroid"
    ycol = "y_centroid" if "y_centroid" in tbl.colnames else "ycentroid"
    H, W = data.shape
    sigs = []
    for row in list(tbl)[:n_stars]:
        x, y = int(row[xcol]), int(row[ycol])
        if x < stamp or y < stamp or x > W - stamp or y > H - stamp:
            continue
        st = data[y - stamp:y + stamp + 1, x - stamp:x + stamp + 1]
        lam_maj, lam_min, _ = measure_psf_tensor(st)
        # round-star check: use the geometric-mean sigma as the PSF width
        if lam_min > 0 and lam_maj / max(lam_min, 1e-6) < 1.5:
            sigs.append(math.sqrt(math.sqrt(lam_maj * lam_min)))
    if not sigs:
        return fwhm_px
    return float(np.median(sigs)) * 2.3548


def stellar_psf_anisotropy(image: np.ndarray, *, fwhm_px: float = 4.0,
                              n_stars: int = 60, stamp: int = 18,
                              min_snr: float = 20.0) -> float:
    """Median (lambda_major - lambda_minor) of bright stars = the
    instrumental PSF anisotropy floor to subtract from a trail measurement.
    Returns 0.0 if it cannot be measured."""
    try:
        from photutils.detection import DAOStarFinder
        from astropy.stats import sigma_clipped_stats
    except ImportError:
        return 0.0
    data = np.asarray(image, float)
    _m, med, std = sigma_clipped_stats(data, sigma=3.0)
    tbl = DAOStarFinder(fwhm=fwhm_px, threshold=min_snr * std)(data - med)
    if tbl is None or len(tbl) == 0:
        return 0.0
    xcol = "x_centroid" if "x_centroid" in tbl.colnames else "xcentroid"
    ycol = "y_centroid" if "y_centroid" in tbl.colnames else "ycentroid"
    H, W = data.shape
    aniso = []
    for row in list(tbl)[:n_stars]:
        x, y = int(row[xcol]), int(row[ycol])
        if x < stamp or y < stamp or x > W - stamp or y > H - stamp:
            continue
        st = data[y - stamp:y + stamp + 1, x - stamp:x + stamp + 1]
        lam_maj, lam_min, _ = measure_psf_tensor(st)
        if lam_min > 0:
            aniso.append(lam_maj - lam_min)
    return float(np.median(aniso)) if aniso else 0.0


def rate_from_stamp(stamp: np.ndarray, *, psf_aniso: float = 0.0,
                      pixscale_arcsec: float = 0.263, t_exp_s: float = 90.0,
                      wcs=None, x_pix: float | None = None,
                      y_pix: float | None = None) -> RateEstimate:
    """Estimate the angular rate + motion PA from one source's trailed PSF.

    psf_aniso: instrumental (lambda_maj - lambda_min) from `stellar_psf_anisotropy`,
               subtracted so only the trail's elongation contributes.
    wcs/x_pix/y_pix: if given, the pixel-frame PA is rotated to sky PA using
               the local WCS orientation.
    """
    lam_maj, lam_min, theta = measure_psf_tensor(stamp)
    excess = (lam_maj - lam_min) - psf_aniso
    L_px = math.sqrt(12.0 * excess) if excess > 0 else 0.0
    rate = L_px * pixscale_arcsec / t_exp_s * 3600.0   # arcsec/hr
    finite = stamp[np.isfinite(stamp)]
    if finite.size == 0:
        return RateEstimate(rate_arcsec_hr=0.0, pa_deg_sky=None, pa_deg_pix=0.0,
                              trail_px=0.0, snr=0.0, lambda_major=0.0,
                              lambda_minor=0.0)
    bg = float(np.median(finite))
    sub = np.clip(finite - bg, 0, None)
    lo = finite[finite < np.percentile(finite, 84)]
    noise = (np.std(lo) if lo.size else np.std(finite)) + 1e-9
    snr = float(sub.max() / noise)
    # `theta` is the major-axis angle measured CCW from the +x pixel axis.
    # Report the motion PA in the usual convention (from +y, i.e. North-like
    # in the pixel frame): PA = 90 - theta.
    pa_pix = (90.0 - math.degrees(theta)) % 180.0
    pa_sky = None
    if wcs is not None and x_pix is not None and y_pix is not None:
        pa_sky = _pix_pa_to_sky(wcs, x_pix, y_pix, theta) % 180.0
    return RateEstimate(rate_arcsec_hr=rate, pa_deg_sky=pa_sky,
                          pa_deg_pix=pa_pix, trail_px=L_px, snr=snr,
                          lambda_major=lam_maj, lambda_minor=lam_min)


def fit_trailed_psf(stamp: np.ndarray, *, psf_fwhm_px: float = 3.8,
                      pixscale_arcsec: float = 0.263, t_exp_s: float = 90.0,
                      wcs=None, x_pix: float | None = None,
                      y_pix: float | None = None) -> "RateEstimate":
    """Maximum-likelihood trailed-PSF fit -- the SMARTER rate estimator.

    Instead of crude second moments (noise-limited below ~500"/hr on
    90s/1" data), forward-model the source as a Gaussian PSF convolved with
    a uniform line segment of length L at angle theta:

        I(x,y) = bg + amp * exp(-v^2/2σ^2) *
                 ½[erf((L/2-u)/(σ√2)) - erf((-L/2-u)/(σ√2))]

    where (u, v) are the along/cross-track coordinates. This is the exact
    profile of a trailed point source and acts as a MATCHED FILTER -- it
    uses every pixel, so it recovers L (=> rate) and theta far below the
    moment noise floor. σ is fixed from the local PSF (psf_fwhm_px).

    Falls back to the moment estimator if scipy/least_squares is missing or
    the fit fails.
    """
    finite = np.isfinite(stamp)
    if not finite.any():
        return RateEstimate(0.0, None, 0.0, 0.0, 0.0, 0.0, 0.0)
    bg0 = float(np.nanmedian(stamp))
    data = np.where(finite, stamp, bg0).astype(float)
    H, W = data.shape
    yy, xx = np.mgrid[0:H, 0:W]
    sigma = psf_fwhm_px / 2.3548
    sub = np.clip(data - bg0, 0, None)
    tot = sub.sum()
    if tot <= 0:
        return RateEstimate(0.0, None, 0.0, 0.0, 0.0, 0.0, 0.0)
    xc0 = float((sub * xx).sum() / tot)
    yc0 = float((sub * yy).sum() / tot)
    # seed L, theta from moments
    lam_maj, lam_min, theta0 = measure_psf_tensor(data)
    L0 = math.sqrt(max(12.0 * (lam_maj - lam_min), 0.0))
    amp0 = float(sub.max())
    noise = (np.std(data[data < np.percentile(data, 84)]) + 1e-9)
    try:
        from scipy.optimize import least_squares
        from scipy.special import erf
    except ImportError:
        est = rate_from_stamp(stamp, pixscale_arcsec=pixscale_arcsec,
                                t_exp_s=t_exp_s, wcs=wcs, x_pix=x_pix, y_pix=y_pix)
        return est

    def model(p):
        xc, yc, amp, bg, L, th = p
        ct, st = math.cos(th), math.sin(th)
        u = (xx - xc) * st + (yy - yc) * ct        # along-track
        v = -(xx - xc) * ct + (yy - yc) * st       # cross-track
        s2 = sigma * math.sqrt(2.0)
        along = 0.5 * (erf((L / 2 - u) / s2) - erf((-L / 2 - u) / s2))
        cross = np.exp(-v ** 2 / (2 * sigma ** 2))
        return bg + amp * cross * along

    def resid(p):
        return (model(p) - data).ravel() / noise

    p0 = [xc0, yc0, amp0 * max(L0, 1.0), bg0, max(L0, 0.1), theta0]
    lb = [xc0 - 3, yc0 - 3, 0.0, bg0 - 5 * noise, 0.0, theta0 - math.pi]
    ub = [xc0 + 3, yc0 + 3, np.inf, bg0 + 5 * noise, max(W, H), theta0 + math.pi]
    try:
        sol = least_squares(resid, p0, bounds=(lb, ub), method="trf",
                              max_nfev=80)
        xc, yc, amp, bg, L, th = sol.x
    except Exception:
        return rate_from_stamp(stamp, pixscale_arcsec=pixscale_arcsec,
                                 t_exp_s=t_exp_s, wcs=wcs, x_pix=x_pix, y_pix=y_pix)
    rate = L * pixscale_arcsec / t_exp_s * 3600.0
    snr = float(np.clip(amp, 0, None) / noise)
    pa_pix = (90.0 - math.degrees(th)) % 180.0
    pa_sky = None
    if wcs is not None and x_pix is not None and y_pix is not None:
        pa_sky = _pix_pa_to_sky(wcs, x_pix, y_pix, th) % 180.0
    return RateEstimate(rate_arcsec_hr=rate, pa_deg_sky=pa_sky,
                          pa_deg_pix=pa_pix, trail_px=L, snr=snr,
                          lambda_major=0.0, lambda_minor=0.0)


def _pix_pa_to_sky(wcs, x, y, theta_pix_rad: float) -> float:
    """Rotate a pixel-frame major-axis angle to a sky PA (deg, E of N).

    Steps one pixel along the major axis, projects both endpoints through
    the WCS, and measures the on-sky bearing. Robust to arbitrary CCD
    rotation/flip in the WCS."""
    # theta is measured CCW from +x, so the major-axis unit vector is
    # (cos theta, sin theta) in (x, y) pixel coordinates.
    dx = math.cos(theta_pix_rad)
    dy = math.sin(theta_pix_rad)
    ra0, dec0 = wcs.pixel_to_world_values(x, y)
    ra1, dec1 = wcs.pixel_to_world_values(x + dx, y + dy)
    cd = math.cos(math.radians(float(dec0)))
    dra = (float(ra1) - float(ra0)) * cd
    ddec = float(dec1) - float(dec0)
    return math.degrees(math.atan2(dra, ddec))
