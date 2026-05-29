"""Long-term (secular) symplectic dynamics -- MASTER_PLAN.md Stage 30.

The short-arc propagators (`ephemeris_nbody.py`, DOP853) are accurate but NOT
symplectic: integrated for millions of years their energy drifts and the cost is
prohibitive. Secular questions -- does a hypothesized distant planet shepherd the
real clustered eTNOs? how does a "with vs without" trajectory difference ACCUMULATE
over time? -- need a structure-preserving integrator. This module implements the
genuine tool used for long-term solar-system dynamics (SWIFT / MERCURY / REBOUND):

  * a universal-variable (Stumpff) Kepler propagator, exact for any eccentricity
    (the real eTNOs reach e ~ 0.93), vectorised over all bodies; and

  * a second-order symplectic Wisdom-Holman map in DEMOCRATIC HELIOCENTRIC
    coordinates (Duncan, Levison & Lee 1998): split the Hamiltonian into a
    Kepler part (each body orbits the Sun's mass), an interaction part
    (mutual planet-planet gravity) and a "solar jump" part (the reflex of the
    Sun). Massive bodies and massless test particles are handled together.

Everything is STANDARD Newtonian gravity (the credibility firewall): the
coherence field is never in the dynamics, only ever a scoring layer downstream.

Coordinates (democratic heliocentric, working in the COM frame of the included
bodies so total momentum is zero):
    Q_i = r_i - r_sun            (heliocentric position, km)
    V_i = barycentric velocity   (km/s; the conjugate momentum is m_i V_i)
The Sun is implicit at the origin of Q; its barycentric velocity is recovered
from momentum conservation, v_sun = -(1/GM_sun) * sum_j GM_j V_j.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..data.constants import GM_SUN, AU_KM
from ..data.ephemeris import body_gm, body_state

YEAR_S = 365.25 * 86400.0       # Julian year (s)


# --------------------------------------------------------------------------- #
# Universal-variable Kepler propagator (Stumpff functions)
# --------------------------------------------------------------------------- #
def _stumpff(psi):
    """Stumpff functions c2(psi), c3(psi), vectorised and branch-safe."""
    psi = np.asarray(psi, float)
    c2 = np.empty_like(psi)
    c3 = np.empty_like(psi)

    pos = psi > 1e-8
    neg = psi < -1e-8
    sml = ~(pos | neg)

    sp = np.sqrt(np.where(pos, psi, 1.0))
    c2 = np.where(pos, (1.0 - np.cos(sp)) / np.where(pos, psi, 1.0), c2)
    c3 = np.where(pos, (sp - np.sin(sp)) / np.where(pos, sp ** 3, 1.0), c3)

    sn = np.sqrt(np.where(neg, -psi, 1.0))
    c2 = np.where(neg, (1.0 - np.cosh(sn)) / np.where(neg, psi, 1.0), c2)
    c3 = np.where(neg, (np.sinh(sn) - sn) / np.where(neg, sn ** 3, 1.0), c3)

    # series near zero (avoids 0/0); good to ~1e-12 for |psi| < 1e-8
    c2 = np.where(sml, 0.5 - psi / 24.0 + psi * psi / 720.0, c2)
    c3 = np.where(sml, 1.0 / 6.0 - psi / 120.0 + psi * psi / 5040.0, c3)
    return c2, c3


def kepler_step(R, V, mu, dt, tol=1e-12, max_iter=80):
    """Advance state(s) (R, V) by dt on a Kepler orbit about `mu` (km^3/s^2).

    R, V are (..., 3). Uses the universal-variable formulation with Newton
    iteration on the universal anomaly; exact for elliptic/parabolic/hyperbolic
    orbits. Returns (R_new, V_new) of the same shape.
    """
    R = np.asarray(R, float)
    V = np.asarray(V, float)
    single = R.ndim == 1
    if single:
        R = R[None, :]
        V = V[None, :]

    sqrtmu = math.sqrt(mu)
    r0 = np.linalg.norm(R, axis=1)
    v0sq = np.einsum("ij,ij->i", V, V)
    rdotv = np.einsum("ij,ij->i", R, V)
    u = rdotv / sqrtmu                       # = r0 * vr0 / sqrt(mu)
    alpha = 2.0 / r0 - v0sq / mu             # = 1/a

    # initial guess for the universal anomaly chi
    chi = sqrtmu * dt * alpha                # elliptic guess (also fine as a seed)
    hyp = alpha < 0
    if np.any(hyp):
        a = 1.0 / alpha
        num = -2.0 * mu * alpha * dt
        den = rdotv + np.sign(dt) * np.sqrt(np.where(hyp, -mu * a, 1.0)) * (1.0 - r0 * alpha)
        guess = np.sign(dt) * np.sqrt(np.where(hyp, -a, 1.0)) * np.log(
            np.where(hyp, num / np.where(den == 0, 1e-300, den), 1.0))
        chi = np.where(hyp, guess, chi)

    for _ in range(max_iter):
        psi = chi * chi * alpha
        c2, c3 = _stumpff(psi)
        chi2 = chi * chi
        r = chi2 * c2 + u * chi * (1.0 - psi * c3) + r0 * (1.0 - psi * c2)
        f_chi = (chi2 * chi * c3 + u * chi2 * c2 + r0 * chi * (1.0 - psi * c3)
                 - sqrtmu * dt)
        dchi = -f_chi / r
        chi = chi + dchi
        if np.max(np.abs(dchi)) < tol:
            break

    psi = chi * chi * alpha
    c2, c3 = _stumpff(psi)
    chi2 = chi * chi
    f = 1.0 - chi2 / r0 * c2
    g = dt - chi2 * chi / sqrtmu * c3
    Rn = f[:, None] * R + g[:, None] * V
    rn = np.linalg.norm(Rn, axis=1)
    gdot = 1.0 - chi2 / rn * c2
    fdot = sqrtmu / (rn * r0) * chi * (psi * c3 - 1.0)
    Vn = fdot[:, None] * R + gdot[:, None] * V

    if single:
        return Rn[0], Vn[0]
    return Rn, Vn


# --------------------------------------------------------------------------- #
# Orbital elements <-> state (heliocentric, mu = GM_sun by default)
# --------------------------------------------------------------------------- #
def _rot_313(i, Om, om):
    co, so = math.cos(om), math.sin(om)
    cO, sO = math.cos(Om), math.sin(Om)
    ci, si = math.cos(i), math.sin(i)
    return np.array([
        [cO * co - sO * so * ci, -cO * so - sO * co * ci, sO * si],
        [sO * co + cO * so * ci, -sO * so + cO * co * ci, -cO * si],
        [so * si, co * si, ci],
    ])


def elements_to_state(a_au, e, i_deg, Om_deg, om_deg, nu_deg, mu=GM_SUN):
    """Heliocentric state (pos km, vel km/s) from Keplerian elements at true anomaly nu."""
    a = a_au * AU_KM
    i, Om, om, nu = (math.radians(v) for v in (i_deg, Om_deg, om_deg, nu_deg))
    p = a * (1.0 - e * e)
    r = p / (1.0 + e * math.cos(nu))
    r_pf = np.array([r * math.cos(nu), r * math.sin(nu), 0.0])
    s = math.sqrt(mu / p)
    v_pf = np.array([-s * math.sin(nu), s * (e + math.cos(nu)), 0.0])
    Rm = _rot_313(i, Om, om)
    return Rm @ r_pf, Rm @ v_pf


def state_to_elements(r, v, mu=GM_SUN):
    """Osculating elements from a heliocentric state. Returns a dict.

    `varpi` (longitude of perihelion = Omega + omega) and the perihelion-direction
    unit vector are what the clustering analysis uses.
    """
    r = np.asarray(r, float)
    v = np.asarray(v, float)
    rmag = np.linalg.norm(r)
    vmag2 = float(v @ v)
    h = np.cross(r, v)
    hmag = np.linalg.norm(h)
    n = np.cross([0.0, 0.0, 1.0], h)          # node vector
    nmag = np.linalg.norm(n)
    evec = ((vmag2 - mu / rmag) * r - (r @ v) * v) / mu
    e = float(np.linalg.norm(evec))
    energy = vmag2 / 2.0 - mu / rmag
    a = -mu / (2.0 * energy) if abs(energy) > 0 else math.inf
    inc = math.acos(max(-1.0, min(1.0, h[2] / hmag)))
    Om = math.atan2(n[1], n[0]) if nmag > 1e-15 else 0.0
    if e > 1e-12 and nmag > 1e-15:
        om = math.acos(max(-1.0, min(1.0, (n @ evec) / (nmag * e))))
        if evec[2] < 0:
            om = 2 * math.pi - om
    else:
        om = 0.0
    peri_hat = evec / e if e > 1e-12 else r / rmag
    varpi = (Om + om) % (2 * math.pi)
    return {"a_au": a / AU_KM, "e": e, "i_deg": math.degrees(inc),
            "Omega_deg": math.degrees(Om) % 360.0, "omega_deg": math.degrees(om) % 360.0,
            "varpi_deg": math.degrees(varpi), "peri_hat": peri_hat}


# --------------------------------------------------------------------------- #
# System assembly (real DE440 initial conditions)
# --------------------------------------------------------------------------- #
#: The giant planets dominate the secular dynamics of distant objects; the inner
#: planets are negligible for eTNOs (and would only shorten the stable timestep).
GIANTS = ("JUPITER BARYCENTER", "SATURN BARYCENTER",
          "URANUS BARYCENTER", "NEPTUNE BARYCENTER")


@dataclass
class System:
    """A democratic-heliocentric system ready for the symplectic map."""
    Q: np.ndarray          # (N,3) heliocentric positions, massive first then test
    V: np.ndarray          # (N,3) barycentric velocities
    gm: np.ndarray         # (n_massive,) GM of each massive body
    gm0: float             # central (Sun) GM
    n_massive: int
    names: list = field(default_factory=list)

    @property
    def n_test(self):
        return self.Q.shape[0] - self.n_massive

    def v_sun(self):
        """Sun's barycentric velocity, recovered from momentum conservation."""
        Vm = self.V[:self.n_massive]
        return -(self.gm[:, None] * Vm).sum(axis=0) / self.gm0


def build_system(epoch_utc, massive=GIANTS, central="SUN"):
    """Assemble Sun + giant planets from real DE440 states, in the COM frame.

    Returns a System with no test particles yet (add with `add_test_particles`).
    """
    bodies = [central, *massive]
    gms = np.array([body_gm(b) for b in bodies])           # [Sun, giants...]
    from ..data.ephemeris import et as _et
    e0 = _et(epoch_utc)
    states = np.array([body_state(b, e0, "J2000", "SSB") for b in bodies])
    pos_ssb = states[:, :3]
    vel_ssb = states[:, 3:]
    # COM frame of the included bodies (enforces sum of momenta = 0)
    v_com = (gms[:, None] * vel_ssb).sum(axis=0) / gms.sum()
    vel = vel_ssb - v_com
    q_helio = pos_ssb - pos_ssb[0]                          # relative to Sun
    # massive bodies = the giants (Sun is implicit at the origin)
    Q = q_helio[1:].copy()
    V = vel[1:].copy()
    gm = gms[1:].copy()
    sys = System(Q=Q, V=V, gm=gm, gm0=float(gms[0]), n_massive=len(massive),
                 names=list(massive))
    sys._v_sun0 = vel[0].copy()                            # Sun's barycentric vel (for test ICs)
    return sys


def add_massive(sys, name, gm, q_helio, v_helio):
    """Insert an additional massive body (e.g. a hypothesized Planet 9) BEFORE test particles."""
    v_bary = np.asarray(v_helio, float) + sys._v_sun0
    nm = sys.n_massive
    Q = np.vstack([sys.Q[:nm], q_helio, sys.Q[nm:]])
    V = np.vstack([sys.V[:nm], v_bary, sys.V[nm:]])
    out = System(Q=Q, V=V, gm=np.append(sys.gm, gm), gm0=sys.gm0,
                 n_massive=nm + 1, names=[*sys.names, name])
    out._v_sun0 = sys._v_sun0
    return out


def add_test_particles(sys, helio_states):
    """Append massless test particles given heliocentric (pos, vel) pairs."""
    Qt = np.array([s[0] for s in helio_states], float)
    Vt = np.array([s[1] for s in helio_states], float) + sys._v_sun0
    out = System(Q=np.vstack([sys.Q, Qt]), V=np.vstack([sys.V, Vt]),
                 gm=sys.gm.copy(), gm0=sys.gm0, n_massive=sys.n_massive,
                 names=list(sys.names))
    out._v_sun0 = sys._v_sun0
    return out


# --------------------------------------------------------------------------- #
# Symplectic Wisdom-Holman map (democratic heliocentric)
# --------------------------------------------------------------------------- #
def _interaction_accel(Q, gm, n_massive):
    """Mutual gravity among massive bodies + pull of massive bodies on every body.

    The Sun is excluded here (its attraction is in the Kepler drift). Returns
    accelerations (N,3) on all bodies.
    """
    Qm = Q[:n_massive]                                     # (M,3) sources
    diff = Qm[None, :, :] - Q[:, None, :]                  # (N,M,3)
    dist = np.linalg.norm(diff, axis=2)                    # (N,M)
    # mask self-interaction for the massive block
    inv = np.zeros_like(dist)
    nz = dist > 0
    inv[nz] = 1.0 / dist[nz] ** 3
    for i in range(n_massive):
        inv[i, i] = 0.0
    return np.einsum("nm,nmc,m->nc", inv, diff, gm)


def _jump_shift(V, gm, gm0, n_massive, factor):
    """Democratic-heliocentric 'solar jump': common shift of all positions."""
    Vm = V[:n_massive]
    return factor * (gm[:, None] * Vm).sum(axis=0) / gm0


def step(sys, dt):
    """One second-order symplectic step (half kick - half jump - drift - half jump - half kick)."""
    Q, V = sys.Q, sys.V
    nm, gm, gm0 = sys.n_massive, sys.gm, sys.gm0

    V = V + 0.5 * dt * _interaction_accel(Q, gm, nm)        # half kick
    Q = Q + _jump_shift(V, gm, gm0, nm, 0.5 * dt)           # half jump
    Q, V = kepler_step(Q, V, gm0, dt)                       # full Kepler drift
    Q = Q + _jump_shift(V, gm, gm0, nm, 0.5 * dt)           # half jump
    V = V + 0.5 * dt * _interaction_accel(Q, gm, nm)        # half kick

    sys.Q, sys.V = Q, V
    return sys


def energy(sys):
    """Total Hamiltonian (G=1, mass==GM units). For relative-conservation checks only."""
    Q, V, gm, gm0, nm = sys.Q, sys.V, sys.gm, sys.gm0, sys.n_massive
    Qm, Vm = Q[:nm], V[:nm]
    ke = 0.5 * (gm * np.einsum("ij,ij->i", Vm, Vm)).sum()
    sun_pe = -(gm0 * gm / np.linalg.norm(Qm, axis=1)).sum()
    p_tot = (gm[:, None] * Vm).sum(axis=0)
    h_sun = 0.5 * (p_tot @ p_tot) / gm0
    h_int = 0.0
    for i in range(nm):
        for j in range(i + 1, nm):
            h_int -= gm[i] * gm[j] / np.linalg.norm(Qm[i] - Qm[j])
    return float(ke + sun_pe + h_sun + h_int)


def angular_momentum(sys):
    """Barycentric angular-momentum magnitude of the massive bodies (mass==GM units)."""
    Q, V, gm, gm0, nm = sys.Q, sys.V, sys.gm, sys.gm0, sys.n_massive
    Qm, Vm = Q[:nm], V[:nm]
    v_sun = -(gm[:, None] * Vm).sum(axis=0) / gm0           # Sun barycentric vel
    r_sun = -(gm[:, None] * Qm).sum(axis=0) / (gm0 + gm.sum())  # Sun barycentric pos
    L = gm0 * np.cross(r_sun, v_sun)
    for i in range(nm):
        L = L + gm[i] * np.cross(r_sun + Qm[i], Vm[i])
    return float(np.linalg.norm(L))


def integrate(sys, dt, n_steps, record_every=0, record_test_elements=False):
    """Integrate `n_steps` of size `dt` (s). Optionally record diagnostics.

    Returns a dict with the final System and, if requested, recorded time series:
      times (yr), test-particle positions, and/or osculating elements.
    """
    rec_t, rec_qtest, rec_elem, rec_energy = [], [], [], []
    nm = sys.n_massive
    for k in range(n_steps):
        if record_every and (k % record_every == 0):
            rec_t.append(k * dt / YEAR_S)
            rec_qtest.append(sys.Q[nm:].copy())
            rec_energy.append(energy(sys))
            if record_test_elements:
                vs = sys.v_sun()
                rec_elem.append([state_to_elements(sys.Q[nm + j], sys.V[nm + j] - vs)
                                 for j in range(sys.n_test)])
        step(sys, dt)
    out = {"system": sys, "n_steps": n_steps, "dt_s": dt,
           "span_yr": n_steps * dt / YEAR_S}
    if record_every:
        out["times_yr"] = np.array(rec_t)
        out["q_test"] = np.array(rec_qtest)                # (T, n_test, 3)
        out["energy"] = np.array(rec_energy)
        if record_test_elements:
            out["elements"] = rec_elem                     # list[T] of list[n_test] dicts
    return out


# --------------------------------------------------------------------------- #
# Clustering metric (the Planet 9 evidence statistic)
# --------------------------------------------------------------------------- #
def perihelion_resultant(elements_row):
    """Mean resultant length R in [0,1] of the perihelion-direction unit vectors.

    R near 1 => perihelia tightly clustered in space; R near 0 => dispersed.
    This is the spatial-clustering statistic behind the Planet 9 argument.
    """
    hats = np.array([e["peri_hat"] for e in elements_row])
    return float(np.linalg.norm(hats.mean(axis=0)))


def varpi_dispersion_deg(elements_row):
    """Circular standard deviation (deg) of the longitude of perihelion varpi."""
    ang = np.radians([e["varpi_deg"] for e in elements_row])
    C, S = np.cos(ang).mean(), np.sin(ang).mean()
    R = math.hypot(C, S)
    return math.degrees(math.sqrt(-2.0 * math.log(R))) if R > 1e-9 else float("inf")
