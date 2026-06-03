"""Rate-constrained multi-night linker -- the scalable discovery linker.

The all-strategy `discover_in_images_chains` is O(N^2) in tracklets and OOMs on
rich fields (16k tracklets -> crash), and 2-point tracklets are mostly chance
pairs, so its output is dominated by the false floor. This module fixes both:

  1. WITHIN-NIGHT (MOPS-style >=3-point tracks): a real moving object appears
     in >=3 of a night's repeat exposures along a straight, constant-rate line.
     Requiring >=3 collinear points (seed from a consecutive-exposure pair via a
     KD-tree near-neighbour query, then GROW along the predicted line) collapses
     the O(N^2) chance-pair explosion to a small set of mostly-real tracks AND
     yields an accurate within-night rate vector.

  2. CROSS-NIGHT (rate-constrained): extrapolate each within-night track to the
     other nights using its measured rate and search only a TIGHT box (sized by
     the rate uncertainty x gap) via a per-night KD-tree -- O(N log N), not all
     pairs. Link when position AND rate are consistent. Chains spanning >=2
     nights are candidate arcs; chance alignments do not repeat across nights.

All angles are handled on a local tangent plane (arcsec) about the field
centre, which is exact enough for a single DECam field (~2 deg).
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Track:
    """A within-night constant-rate track (>=3 detections)."""
    night: int
    jd_mid: float
    ra_mid: float          # deg
    dec_mid: float         # deg
    vra: float             # arcsec/hr on tangent plane (RA*cos dec direction)
    vdec: float            # arcsec/hr
    rate_arcsec_hr: float
    n_points: int
    mag: float
    point_ids: tuple       # ids of the member detections (for dedup / chains)
    ra0: float = 0.0       # tangent-plane origin
    dec0: float = 0.0
    sources: tuple = ()    # the member detection Source objects (for labelling)


def _project(ra, dec, ra0, dec0):
    """(ra,dec) deg -> tangent-plane (x,y) arcsec about (ra0,dec0)."""
    cd = math.cos(math.radians(dec0))
    x = (np.asarray(ra) - ra0) * cd * 3600.0
    y = (np.asarray(dec) - dec0) * 3600.0
    return x, y


def _deproject(x, y, ra0, dec0):
    cd = math.cos(math.radians(dec0))
    ra = ra0 + (x / 3600.0) / cd
    dec = dec0 + y / 3600.0
    return ra, dec


def build_within_night_tracks(sources, *, min_rate_arcsec_hr=2.0,
                               max_rate_arcsec_hr=120.0, pos_tol_arcsec=2.5,
                               min_points=3):
    """Build >=min_points constant-rate tracks from one night's Sources.

    `sources` all belong to one night; each has .ra/.dec (deg), .mjd (exposure
    epoch), .mag. Returns a list of Track. KD-tree seeded + grown, so it scales.
    """
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        cKDTree = None
    if len(sources) < min_points:
        return []
    ra0 = float(np.median([s.ra for s in sources]))
    dec0 = float(np.median([s.dec for s in sources]))
    # group by exposure epoch
    by_exp = defaultdict(list)
    for s in sources:
        by_exp[round(s.mjd, 6)].append(s)
    epochs = sorted(by_exp)
    if len(epochs) < min_points:
        return []
    # per-exposure projected coords + KD-tree
    exp_xy = {}
    exp_trees = {}
    for e in epochs:
        ss = by_exp[e]
        x, y = _project(np.array([s.ra for s in ss]), np.array([s.dec for s in ss]), ra0, dec0)
        xy = np.column_stack([x, y])
        exp_xy[e] = (ss, xy)
        if cKDTree is not None and len(xy):
            exp_trees[e] = cKDTree(xy)

    def neighbours(e, pt, r):
        ss, xy = exp_xy[e]
        if e in exp_trees:
            idx = exp_trees[e].query_ball_point(pt, r)
            return [(i, ss[i], xy[i]) for i in idx]
        d = np.hypot(xy[:, 0] - pt[0], xy[:, 1] - pt[1]) if len(xy) else np.array([])
        return [(i, ss[i], xy[i]) for i in np.where(d <= r)[0]]

    tracks = []
    seen = set()
    # seed from each consecutive-exposure pair, grow along the line
    for ia in range(len(epochs) - 1):
        ea = epochs[ia]
        ss_a, xy_a = exp_xy[ea]
        for ib in range(ia + 1, min(ia + 3, len(epochs))):   # seed within next 2 exposures
            eb = epochs[ib]
            dt_hr = (eb - ea) * 24.0
            if dt_hr <= 0:
                continue
            rmax = max_rate_arcsec_hr * dt_hr + pos_tol_arcsec
            for i, sa, pa in [(i, ss_a[i], xy_a[i]) for i in range(len(ss_a))]:
                for j, sb, pb in neighbours(eb, pa, rmax):
                    dx = pb[0] - pa[0]; dy = pb[1] - pa[1]
                    rate = math.hypot(dx, dy) / dt_hr
                    if not (min_rate_arcsec_hr <= rate <= max_rate_arcsec_hr):
                        continue
                    vra = dx / dt_hr; vdec = dy / dt_hr      # arcsec/hr
                    # GROW: predict at every epoch, collect points within tol
                    pts = [(ea, i, sa, pa), (eb, j, sb, pb)]
                    for ec in epochs:
                        if ec in (ea, eb):
                            continue
                        dt2 = (ec - ea) * 24.0
                        pred = (pa[0] + vra * dt2, pa[1] + vdec * dt2)
                        nb = neighbours(ec, pred, pos_tol_arcsec)
                        if nb:
                            nb.sort(key=lambda t: (t[2][0] - pred[0]) ** 2 + (t[2][1] - pred[1]) ** 2)
                            k, sc, pc = nb[0]
                            pts.append((ec, k, sc, pc))
                    if len(pts) < min_points:
                        continue
                    key = tuple(sorted(id(p[2]) for p in pts))
                    if key in seen:
                        continue
                    seen.add(key)
                    # least-squares refine velocity over all points
                    t0 = pts[0][0]
                    th = np.array([(p[0] - t0) * 24.0 for p in pts])
                    xs = np.array([p[3][0] for p in pts]); ys = np.array([p[3][1] for p in pts])
                    A = np.column_stack([np.ones_like(th), th])
                    (bx, mx), *_ = np.linalg.lstsq(A, xs, rcond=None)
                    (by, my), *_ = np.linalg.lstsq(A, ys, rcond=None)
                    thm = float(th.mean())
                    xm = bx + mx * thm; ym = by + my * thm
                    ram, decm = _deproject(xm, ym, ra0, dec0)
                    night = int(round(pts[0][2].mjd))
                    jd_mid = float(np.mean([p[2].mjd for p in pts])) + 2400000.5
                    mag = float(np.median([p[2].mag for p in pts if p[2].mag > -50] or [-99]))
                    tracks.append(Track(
                        night=night, jd_mid=jd_mid, ra_mid=float(ram), dec_mid=float(decm),
                        vra=float(mx), vdec=float(my),
                        rate_arcsec_hr=float(math.hypot(mx, my)),
                        n_points=len(pts), mag=mag,
                        point_ids=key, ra0=ra0, dec0=dec0,
                        sources=tuple(p[2] for p in pts)))
    return tracks


def link_rate_constrained(tracks, *, max_rate_change_frac=0.35,
                          base_tol_arcsec=8.0, rate_tol_frac=0.10,
                          max_heading_change_deg=20.0, min_nights=2):
    """Link within-night tracks across nights by rate extrapolation + tight box.

    For a night-i track, extrapolate to night j using its rate vector; accept a
    night-j track whose centroid is within (base_tol + rate_tol_frac * motion)
    and whose rate magnitude + heading are consistent. Greedy transitive chain.
    Returns chains (lists of Track) spanning >= min_nights.
    """
    if not tracks:
        return []
    by_night = defaultdict(list)
    for t in tracks:
        by_night[t.night].append(t)
    nights = sorted(by_night)
    ra0 = tracks[0].ra0; dec0 = tracks[0].dec0
    # per-night tangent-plane positions + KD-tree
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        cKDTree = None
    night_xy = {}; night_tree = {}
    for n in nights:
        ts = by_night[n]
        x, y = _project(np.array([t.ra_mid for t in ts]),
                        np.array([t.dec_mid for t in ts]), ra0, dec0)
        xy = np.column_stack([x, y]); night_xy[n] = (ts, xy)
        if cKDTree is not None and len(xy):
            night_tree[n] = cKDTree(xy)

    def consistent(ta, tb):
        ra_ratio = abs(tb.rate_arcsec_hr - ta.rate_arcsec_hr) / max(ta.rate_arcsec_hr, 1e-6)
        if ra_ratio > max_rate_change_frac:
            return False
        ha = math.degrees(math.atan2(ta.vdec, ta.vra))
        hb = math.degrees(math.atan2(tb.vdec, tb.vra))
        dh = abs((ha - hb + 180) % 360 - 180)
        return dh <= max_heading_change_deg

    # build links i->j (j later night)
    nxt = defaultdict(list)
    for ii, n in enumerate(nights):
        ts, xy = night_xy[n]
        for a, (t, p) in enumerate(zip(ts, xy)):
            for n2 in nights[ii + 1:]:
                dt_hr = (n2 - n) * 24.0
                pred = (p[0] + t.vra * dt_hr, p[1] + t.vdec * dt_hr)
                motion = t.rate_arcsec_hr * dt_hr
                tol = base_tol_arcsec + rate_tol_frac * motion
                ts2, xy2 = night_xy[n2]
                if n2 in night_tree:
                    cand = night_tree[n2].query_ball_point(pred, tol)
                else:
                    d = np.hypot(xy2[:, 0] - pred[0], xy2[:, 1] - pred[1]) if len(xy2) else np.array([])
                    cand = list(np.where(d <= tol)[0])
                for b in cand:
                    if consistent(t, ts2[b]):
                        nxt[id(t)].append(ts2[b])
                if nxt[id(t)]:
                    break        # link to the nearest following night only

    # greedy transitive chains
    chains = []
    used = set()
    for n in nights:
        for t in by_night[n]:
            if id(t) in used:
                continue
            chain = [t]; used.add(id(t)); cur = t
            while nxt.get(id(cur)):
                nextt = min(nxt[id(cur)],
                            key=lambda x: abs(x.rate_arcsec_hr - cur.rate_arcsec_hr))
                if id(nextt) in used:
                    break
                chain.append(nextt); used.add(id(nextt)); cur = nextt
            if len({c.night for c in chain}) >= min_nights:
                chains.append(chain)
    return chains


def coherence_field_tracks(ra, dec, mjd, *, max_rate_arcsec_hr=60.0,
                           cluster_tol_arcsec=2.5, min_nights=3, min_points=3,
                           max_resid_arcsec=1.2, max_night_gap_days=20.0,
                           pair_cap=400000):
    """Find moving objects as COHERENT PEAKS of a field sourced by ALL detections.

    The deep Equation-of-One application: instead of thresholding pairwise links,
    every viable inter-night detection PAIR sources an *exact* rate hypothesis
    (vra, vdec) = (pos_j - pos_i)/(t_j - t_i) -- no rate grid, so no quantization
    wall. At that rate, every detection is back-projected to a common reference
    epoch: x_ref = x - vra*(t-t0). For a real object's true rate, ALL of its
    detections -- across every night, however faint -- collapse to ONE reference
    position (a coherent attractor / field peak); unrelated detections stay
    incoherently spread. A peak drawing detections from >=min_nights distinct
    nights and >=min_points total is a candidate track. Different pairs of the
    SAME object propose the SAME rate, so its peak is reinforced; the final set
    is deduped by membership overlap.

    Crucially this needs the object detected only ONCE per night on >=min_nights
    nights -- it never needs a within-night tracklet -- so it recovers faint
    objects the pairwise tracker (which requires >=3 within-night detections)
    cannot. Returns [{idx, vra, vdec, n_nights, n_points}], best-first.
    """
    try:
        from scipy.spatial import cKDTree
    except Exception:
        cKDTree = None
    ra = np.asarray(ra, float); dec = np.asarray(dec, float); mjd = np.asarray(mjd, float)
    n = len(ra)
    if n < min_points:
        return []
    ra0 = float(np.median(ra)); dec0 = float(np.median(dec)); t0 = float(np.median(mjd))
    cd = math.cos(math.radians(dec0))
    x = (ra - ra0) * cd * 3600.0
    y = (dec - dec0) * 3600.0
    th = (mjd - t0) * 24.0                      # hours from reference epoch
    night = np.floor(mjd - 0.5).astype(int)
    uniq_nights = sorted(set(int(v) for v in night))
    by_night = {u: np.where(night == u)[0] for u in uniq_nights}
    tol2 = cluster_tol_arcsec ** 2

    # 1. enumerate viable inter-night pairs -> exact rate hypotheses. Prune by a
    #    max-rate radius: a partner on a later night must lie within max_rate*dt.
    rates = []                                  # (vra, vdec, anchor_idx)
    capped = False
    for a_i, ua in enumerate(uniq_nights):
        ia = by_night[ua]
        pa = np.column_stack([x[ia], y[ia]])
        for ub in uniq_nights[a_i + 1:]:
            dt_hr = (float(np.median(mjd[by_night[ub]])) - float(np.median(mjd[ia]))) * 24.0
            if abs(dt_hr) > max_night_gap_days * 24.0 or abs(dt_hr) < 1e-6:
                continue
            ib = by_night[ub]
            pb = np.column_stack([x[ib], y[ib]])
            radius = max_rate_arcsec_hr * abs(dt_hr)
            if cKDTree is not None:
                for ka, kb_list in enumerate(cKDTree(pa).query_ball_tree(cKDTree(pb), radius)):
                    for kb in kb_list:
                        dthr = th[ib[kb]] - th[ia[ka]]
                        rates.append(((x[ib[kb]] - x[ia[ka]]) / dthr,
                                      (y[ib[kb]] - y[ia[ka]]) / dthr, int(ia[ka])))
            else:
                for ka in range(len(ia)):
                    d = pb - pa[ka]
                    for kb in np.where(d[:, 0]**2 + d[:, 1]**2 <= radius**2)[0]:
                        dthr = th[ib[kb]] - th[ia[ka]]
                        rates.append(((x[ib[kb]] - x[ia[ka]]) / dthr,
                                      (y[ib[kb]] - y[ia[ka]]) / dthr, int(ia[ka])))
            if len(rates) > pair_cap:
                capped = True; break
        if capped:
            break

    # 2. for each rate hypothesis, collapse ALL detections to t0 and gather the
    #    coherent peak (within tol of the anchor's projected reference position).
    emitted = set(); tracks = []
    for vra, vdec, ai in rates:
        rx = x - vra * th
        ry = y - vdec * th
        members = np.where((rx - rx[ai])**2 + (ry - ry[ai])**2 <= tol2)[0]
        if len(members) < min_points:
            continue
        mn = set(int(night[m]) for m in members)
        if len(mn) < min_nights:
            continue
        key = frozenset(int(m) for m in members)
        if key in emitted:
            continue
        emitted.add(key)
        # COHERENCE ENERGY: a true object is ONE linear motion (x0+vx t, y0+vy t).
        # Refit a single motion to all members; the RMS residual IS the
        # incoherence energy. Chance triples gathered at a fluke rate do not share
        # one velocity across >=3 nights, so their residual blows up -> rejected.
        mt = th[members]
        if float(np.ptp(mt)) > 1e-6:
            A = np.column_stack([np.ones(len(members)), mt])
            cx, *_ = np.linalg.lstsq(A, x[members], rcond=None)
            cy, *_ = np.linalg.lstsq(A, y[members], rcond=None)
            resid = float(np.sqrt(np.mean((x[members] - A @ cx)**2 + (y[members] - A @ cy)**2)))
            vra, vdec = float(cx[1]), float(cy[1])
        else:
            resid = 0.0
        if resid > max_resid_arcsec:
            continue
        tracks.append({"idx": sorted(int(m) for m in members), "vra": vra, "vdec": vdec,
                       "n_nights": len(mn), "n_points": int(len(members)), "resid": resid})

    # 3. dedup: the same object is proposed by many pairs -> keep the most coherent
    #    (most nights, then lowest residual), drop those overlapping it >=50%.
    tracks.sort(key=lambda t: (-t["n_nights"], t["resid"], -t["n_points"]))
    kept = []
    for t in tracks:
        s = set(t["idx"])
        if not any(len(s & set(k["idx"])) >= 0.5 * min(len(s), len(k["idx"])) for k in kept):
            kept.append(t)
    return kept


def link_coherence(tracks, *, base_tol_arcsec=8.0, rate_tol_frac=0.12,
                   sig_rate_frac=0.25, sig_head_deg=20.0, max_energy=1.5,
                   w_rate=0.5, w_head=0.5, min_nights=2):
    """Cross-night linking by Equation-of-One energy minimisation.

    For each night-i track, extrapolate to a later night and score every
    candidate linkage by its total incoherence energy:

        E = E_align(position residual) + w_rate*coh(rate) + w_head*coh(heading)

    where E_align = 1 - exp(-(d/tol)^2) is the alignment kernel (tol grows with
    the extrapolated motion) and the rate/heading terms are standardized
    coherence divergences. Link to the MOST COHERENT (min-E) partner below
    max_energy. This replaces the hard-threshold greedy rule (link_rate_constrained)
    with the unified coherence objective -- the same Equation-of-One selector
    that took variable typing 74% -> 89%, now applied to the discovery linker.
    """
    from .coherence_field import alignment_energy
    if not tracks:
        return []
    by_night = defaultdict(list)
    for t in tracks:
        by_night[t.night].append(t)
    nights = sorted(by_night)
    ra0 = tracks[0].ra0; dec0 = tracks[0].dec0
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        cKDTree = None
    night_xy = {}; night_tree = {}
    for n in nights:
        ts = by_night[n]
        x, y = _project(np.array([t.ra_mid for t in ts]),
                        np.array([t.dec_mid for t in ts]), ra0, dec0)
        xy = np.column_stack([x, y]); night_xy[n] = (ts, xy)
        if cKDTree is not None and len(xy):
            night_tree[n] = cKDTree(xy)

    best = {}     # id(track) -> most-coherent following-night partner
    for ii, n in enumerate(nights):
        ts, xy = night_xy[n]
        for t, p in zip(ts, xy):
            for n2 in nights[ii + 1:]:
                dt_hr = (n2 - n) * 24.0
                pred = (p[0] + t.vra * dt_hr, p[1] + t.vdec * dt_hr)
                motion = t.rate_arcsec_hr * dt_hr
                tol = base_tol_arcsec + rate_tol_frac * motion
                ts2, xy2 = night_xy[n2]
                if n2 in night_tree:
                    cand = night_tree[n2].query_ball_point(pred, 3.0 * tol)  # wide; energy filters
                else:
                    d = np.hypot(xy2[:, 0] - pred[0], xy2[:, 1] - pred[1]) if len(xy2) else np.array([])
                    cand = list(np.where(d <= 3.0 * tol)[0])
                bestE = math.inf; bestc = None
                for b in cand:
                    c = ts2[b]
                    d = math.hypot(xy2[b][0] - pred[0], xy2[b][1] - pred[1])
                    E_pos = alignment_energy(d, 0.0, max(tol, 1.0) ** 2)
                    rate_mis = abs(c.rate_arcsec_hr - t.rate_arcsec_hr) / max(t.rate_arcsec_hr, 1e-6)
                    ha = math.degrees(math.atan2(t.vdec, t.vra))
                    hb = math.degrees(math.atan2(c.vdec, c.vra))
                    dh = abs((ha - hb + 180) % 360 - 180)
                    E = (E_pos + w_rate * (rate_mis / sig_rate_frac) ** 2
                         + w_head * (dh / sig_head_deg) ** 2)
                    if E < bestE:
                        bestE = E; bestc = c
                if bestc is not None and bestE < max_energy:
                    best[id(t)] = bestc
                    break        # nearest following night only
    # transitive chains via the most-coherent links
    chains = []; used = set()
    for n in nights:
        for t in by_night[n]:
            if id(t) in used:
                continue
            chain = [t]; used.add(id(t)); cur = t
            while id(cur) in best:
                nxt = best[id(cur)]
                if id(nxt) in used:
                    break
                chain.append(nxt); used.add(id(nxt)); cur = nxt
            if len({c.night for c in chain}) >= min_nights:
                chains.append(chain)
    return chains
