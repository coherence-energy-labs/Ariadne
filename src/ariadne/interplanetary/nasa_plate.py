"""NASA-grade matplotlib styling for Ariadne mission plates and atlases.

Style decisions derived from JPL Solar System Dynamics and NASA Goddard
mission-design publication graphics:

  * Dark deep-space background (#06090d) with soft cyan grid (#1d2a3a).
  * Bold uppercase mission-style titles in sans-serif; monospaced numerics
    with consistent character width.
  * Perceptually-uniform color-blind-safe palette (cividis as primary,
    plasma as alternate for risk/secondary scales).
  * High-contrast halo'd text on dense backgrounds (labels stay legible
    on viridis/cividis without darkening the cells).
  * Mission-metadata footer band: mission ID, certificate, fidelity,
    generation timestamp. Clear provenance on every emitted image.
  * Trajectory styling: planet orbits as thin dashed circles, mission
    routes as gradient-shaded solid lines with directional arrows.
  * Inset diagrams clearly separated by a 1-px border.

Use:

    from ariadne.interplanetary.nasa_plate import apply_style, mission_footer

    apply_style()
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(...)
    mission_footer(fig, mission_id="EARTH->MARS", cert="abc12345",
                   fidelity="patched_conic_real_ephemerides")
    fig.savefig(...)
"""
from __future__ import annotations

from datetime import datetime
from typing import Iterable

# Color palette --------------------------------------------------------------
DEEP_SPACE_BG = "#06090d"       # near-black with subtle blue tint
PANEL_BG = "#0d141c"            # slightly lighter for panel surfaces
GRID_LINE = "#1d2a3a"           # cyan-tinted grid
GRID_MINOR = "#101820"
TEXT_PRIMARY = "#e6edf3"        # nearly white, easy on the eye
TEXT_SECONDARY = "#8b949e"      # medium grey for labels
TEXT_FAINT = "#5a636d"          # captions / metadata
ACCENT_CYAN = "#58a6ff"         # primary accent
ACCENT_GOLD = "#f0c674"         # secondary accent (warning / minima)
ACCENT_GREEN = "#56d364"
ACCENT_RED = "#f85149"

# Per-role colors used by the navigator (high contrast on dark bg)
ROLE_COLORS = {
    "fastest": "#ff6b9d",       # magenta
    "cheapest": ACCENT_CYAN,    # cyan
    "balanced": ACCENT_GREEN,   # green
    "minimum": ACCENT_GOLD,     # gold
    "pareto": "#a371f7",        # purple
}


def apply_style():
    """Set matplotlib rcParams to NASA-grade defaults. Call once per render."""
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.facecolor": DEEP_SPACE_BG,
        "axes.facecolor": PANEL_BG,
        "axes.edgecolor": GRID_LINE,
        "axes.labelcolor": TEXT_SECONDARY,
        "axes.titlecolor": TEXT_PRIMARY,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlepad": 12,
        "axes.labelsize": 9.5,
        "axes.labelweight": "regular",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "grid.color": GRID_MINOR,
        "grid.alpha": 0.6,
        "grid.linestyle": "-",
        "grid.linewidth": 0.5,
        "text.color": TEXT_PRIMARY,
        "font.family": ["DejaVu Sans"],
        "font.size": 9.5,
        "legend.facecolor": PANEL_BG,
        "legend.edgecolor": GRID_LINE,
        "legend.framealpha": 0.92,
        "legend.fontsize": 8.5,
        "legend.title_fontsize": 9,
        "legend.borderpad": 0.6,
        "lines.linewidth": 1.5,
        "image.cmap": "cividis",         # color-blind safe primary
        "savefig.facecolor": DEEP_SPACE_BG,
        "savefig.edgecolor": "none",
    })


def reset_style():
    """Reset to matplotlib defaults; call when leaving Ariadne plotting."""
    import matplotlib as mpl
    mpl.rcParams.update(mpl.rcParamsDefault)


def halo_text(ax, x, y, text, *, color=TEXT_PRIMARY,
               halo_color=DEEP_SPACE_BG, halo_width=2.0,
               fontsize=8.5, weight="regular", **kwargs):
    """Draw text with a contrasting outline halo for legibility on dense bgs."""
    import matplotlib.patheffects as pe
    return ax.text(x, y, text, color=color, fontsize=fontsize, weight=weight,
                    path_effects=[
                        pe.Stroke(linewidth=halo_width, foreground=halo_color),
                        pe.Normal(),
                    ], **kwargs)


def mission_footer(fig, *, mission_id: str, cert: str | None = None,
                    fidelity: str | None = None,
                    extras: Iterable[tuple[str, str]] = (),
                    timestamp: str | None = None) -> None:
    """Draw the standard mission-metadata footer band at the bottom of `fig`.

    Includes: mission ID (LEFT), certificate hash (CENTER if provided),
    fidelity tier + extras (RIGHT), generation timestamp (FAR RIGHT).
    """
    if timestamp is None:
        # Cannot use Date.now() in some environments; caller can override
        try:
            timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        except Exception:
            timestamp = ""

    # Left -- mission ID
    fig.text(0.012, 0.012, f"ARIADNE / {mission_id.upper()}",
              fontsize=7.5, weight="bold", color=ACCENT_CYAN,
              family="monospace", ha="left", va="bottom")

    # Center -- certificate hash
    if cert:
        fig.text(0.5, 0.012, f"CERT {cert[:24]}",
                  fontsize=7.0, color=TEXT_FAINT, family="monospace",
                  ha="center", va="bottom")

    # Right -- fidelity + extras + timestamp
    bits = []
    if fidelity:
        bits.append(fidelity.replace("_", " ").upper())
    for k, v in extras:
        bits.append(f"{k.upper()}: {v}")
    if timestamp:
        bits.append(timestamp)
    if bits:
        fig.text(0.988, 0.012, "  |  ".join(bits),
                  fontsize=7.0, color=TEXT_FAINT, family="monospace",
                  ha="right", va="bottom")


def mission_title(fig, *, title: str, subtitle: str | None = None):
    """Draw a NASA-style title (large bold uppercase) + subtitle row."""
    fig.text(0.012, 0.978, title.upper(),
              fontsize=14, weight="bold", color=TEXT_PRIMARY,
              family="DejaVu Sans", ha="left", va="top")
    if subtitle:
        fig.text(0.012, 0.951, subtitle,
                  fontsize=9.5, color=TEXT_SECONDARY,
                  family="DejaVu Sans", ha="left", va="top",
                  style="italic")


def style_axes(ax, *, title: str | None = None,
               xlabel: str | None = None, ylabel: str | None = None,
               grid: bool = True):
    """Apply NASA-grade styling to a single ax (after plot data is drawn)."""
    if title:
        ax.set_title(title.upper(), loc="left", pad=10,
                       color=TEXT_PRIMARY, weight="bold", fontsize=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=TEXT_SECONDARY, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT_SECONDARY, fontsize=9)
    if grid:
        ax.grid(True, which="major", color=GRID_LINE, alpha=0.5,
                 linestyle="-", linewidth=0.5)
        ax.grid(True, which="minor", color=GRID_MINOR, alpha=0.3,
                 linestyle=":", linewidth=0.3)
        ax.minorticks_on()
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color(GRID_LINE)
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(colors=TEXT_SECONDARY, which="both")


def card_box(ax, x, y, text: str, *, width=0.30, fontsize=8.0,
              color=TEXT_PRIMARY, bg=PANEL_BG, border=GRID_LINE,
              header: str | None = None):
    """Draw a NASA-style metadata card with optional uppercase header."""
    from matplotlib.patches import FancyBboxPatch

    # Background rounded box
    n_lines = text.count("\n") + 1 + (2 if header else 0)
    # Approximate per-line height in axes coords (fontsize-scaled)
    line_h_axes = max(0.030, fontsize / 220.0)
    height = line_h_axes * n_lines + 0.025
    box = FancyBboxPatch(
        (x, y - height), width, height,
        boxstyle="round,pad=0.012,rounding_size=0.008",
        linewidth=0.7, facecolor=bg, edgecolor=border,
        transform=ax.transAxes, zorder=2,
    )
    ax.add_patch(box)
    if header:
        ax.text(x + 0.014, y - 0.014, header.upper(),
                 fontsize=fontsize - 0.5, weight="bold", color=border,
                 family="monospace", transform=ax.transAxes,
                 va="top", ha="left", zorder=3)
        ax.text(x + 0.014, y - 0.014 - line_h_axes * 1.5, text,
                 fontsize=fontsize, color=color,
                 family="monospace", transform=ax.transAxes,
                 va="top", ha="left", zorder=3,
                 linespacing=1.35)
    else:
        ax.text(x + 0.014, y - 0.014, text,
                 fontsize=fontsize, color=color,
                 family="monospace", transform=ax.transAxes,
                 va="top", ha="left", zorder=3,
                 linespacing=1.35)


def directional_route(ax, xs, ys, *, color, lw=2.0, n_arrows=3,
                       arrow_size=8, alpha=0.95):
    """Draw a route with subtle directional arrows along its length."""
    import numpy as np
    ax.plot(xs, ys, color=color, lw=lw, alpha=alpha, zorder=2)
    if len(xs) < 2 or n_arrows < 1:
        return
    # Place arrows at evenly-spaced fractions along the path
    pts = np.column_stack([xs, ys])
    segs = np.diff(pts, axis=0)
    lens = np.hypot(segs[:, 0], segs[:, 1])
    cum = np.concatenate([[0], np.cumsum(lens)])
    total = cum[-1]
    if total <= 0:
        return
    fracs = np.linspace(0.25, 0.85, n_arrows)
    for f in fracs:
        target = f * total
        # Find segment containing this fraction
        i = max(0, min(len(lens) - 1, int(np.searchsorted(cum, target) - 1)))
        if lens[i] <= 0:
            continue
        u = (target - cum[i]) / lens[i]
        pos = pts[i] + u * segs[i]
        d = segs[i] / lens[i]
        ax.annotate("", xy=pos + d * 1e-3, xytext=pos - d * 1e-3,
                     arrowprops={"arrowstyle": f"-|>", "color": color,
                                  "lw": 0, "alpha": alpha, "mutation_scale": arrow_size},
                     zorder=3)
