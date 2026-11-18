"""Publication figure toolkit: one place for every figure convention.

Everything that a journal copy editor would otherwise fix by hand lives here:
column widths, font sizes, panel letters, scale bars, north arrows, and the
vector export. Figures are built from data files, never drawn by hand, so a
change in the numbers regenerates every panel at once.

Conventions follow the Bulletin of Engineering Geology and the Environment
(Springer) artwork guidelines: single-column width 174 mm is the ceiling, the
text block of this manuscript is 158 mm, lettering sans serif (Arial) at 8 pt or
more, lines at least 0.3 pt, RGB, vector PDF with embedded fonts plus a 600 dpi
PNG for review.
"""

from __future__ import annotations

import io
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrow, Polygon, Rectangle

MM = 1 / 25.4
COL1 = 84 * MM      # journal single column (84 mm)
COL2 = 158 * MM     # this manuscript's text block; journal ceiling is 174 mm
MAXW_MM = 158.0     # artwork is fitted to this, not drawn and left to Word
MAXH_MM = 200.0     # leaves room for the caption on an A4 page

# Muted, print-safe palette. Blues for background, warm for hazard.
C = {
    "ink": "#1F1F1F",
    "muted": "#6B6B6B",
    "grid": "#E3E0D8",
    "shaft": "#C4622D",
    "shaft_off": "#9A9A94",
    "chain_a": "#1B6B4A",
    "chain_b": "#C98A18",
    "chain_c": "#8E3B3B",
    "road1": "#2B4C6F",
    "road2": "#4A7093",
    "road3": "#7C93A8",
    "cavity": "#A33B2D",
    "accent": "#1F6F8B",
    "good": "#1B6B4A",
}


def use_style(lang: str = "en"):
    """Set rcParams once. lang only affects the CJK fallback chain."""
    family = ["Arial", "Helvetica", "DejaVu Sans"]
    if lang == "zh":
        family = ["Microsoft YaHei", "SimHei"] + family
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": family,
        # mathtext otherwise falls back to DejaVu, which is what put two
        # typefaces in the same panel: an alpha from the maths font next to a
        # Latin label from Arial. Springer asks for one sans serif throughout.
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "mathtext.sf": "Arial",
        "mathtext.default": "regular",
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8.5,
        # 8 pt is the journal floor; 7.2 pt was below it once the artwork was
        # placed at 158 mm
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.6,
        "axes.edgecolor": C["ink"],
        "axes.labelcolor": C["ink"],
        "text.color": C["ink"],
        "xtick.color": C["ink"],
        "ytick.color": C["ink"],
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.4,
        "ytick.major.size": 2.4,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "axes.titlesize": 8,
        "axes.titlepad": 3,
        "legend.handletextpad": 0.5,
        "legend.borderaxespad": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,     # editable text in the PDF
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
    })


def panel(ax, letter: str, dx: float = -0.02, dy: float = 1.03):
    """Bold (a) at the top-left of an axes, in axes coordinates."""
    ax.text(dx, dy, f"({letter})", transform=ax.transAxes, fontsize=8,
            fontweight="bold", ha="right", va="bottom", color=C["ink"])


def _nice_length(target: float) -> float:
    p = 10 ** math.floor(math.log10(max(target, 1e-9)))
    for m in (1, 2, 5, 10):
        if p * m >= target * 0.55:
            return p * m
    return p * 10


def scalebar(ax, span: float, unit: str = "km", frac: float = 0.22,
             loc=(0.06, 0.06), color=None, lw: float = 1.5,
             fontsize: float = 6.6, label: str | None = None):
    """Scale bar of a round number near ``frac`` of the axis span.

    ``span`` and ``unit`` must match the axis units - passing km to an axis
    drawn in metres is how a 1 km bar ends up labelled "1 m". The bar is placed
    in axes coordinates, so it survives zoom and aspect changes.
    """
    color = color or C["ink"]
    L = _nice_length(abs(span) * frac)
    x0, y0 = loc
    L_ax = L / abs(span) if span else frac
    ax.plot([x0, x0 + L_ax], [y0, y0], color=color, linewidth=lw,
            solid_capstyle="butt", transform=ax.transAxes, zorder=20, clip_on=False)
    for xx in (x0, x0 + L_ax):
        ax.plot([xx, xx], [y0, y0 + 0.014], color=color, linewidth=lw,
                transform=ax.transAxes, zorder=20, clip_on=False)
    txt = label if label is not None else f"{L:g} {unit}"
    ax.text(x0 + L_ax, y0 + 0.024, txt, transform=ax.transAxes, ha="right",
            va="bottom", fontsize=fontsize, color=color, zorder=20,
            bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.8))


def north_arrow(ax, loc=(0.94, 0.94), size=0.055, conv_deg: float = 0.0,
                color=None, fontsize: float = 7):
    """North arrow in axes coordinates, pointing at true north.

    ``conv_deg`` is the meridian convergence gamma = (lon - lon0) * sin(lat),
    positive when the grid north falls east of true north. At 89.31 E in UTM 45N
    (central meridian 87 E) gamma = +1.57 deg, so true north leans *west* of the
    grid vertical and the arrow is rotated by +gamma; passing a negative value
    tilts it the wrong way, which is easy to miss at this size.
    """
    color = color or C["ink"]
    x, y = loc
    th = math.radians(90 + conv_deg)
    dx, dy = size * math.cos(th), size * 1.6 * math.sin(th)
    ax.annotate("", xy=(x + dx, y + dy), xytext=(x - dx * 0.35, y - dy * 0.35),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=color, linewidth=0.9,
                                mutation_scale=7, shrinkA=0, shrinkB=0), zorder=20)
    ax.text(x + dx * 1.45, y + dy * 1.35, "N", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=fontsize, fontweight="bold", color=color)


def grid(ax, axis="both", alpha: float = 1.0):
    ax.grid(True, axis=axis, color=C["grid"], linewidth=0.4, alpha=alpha)
    ax.set_axisbelow(True)


def fit_extent(ax, xy: np.ndarray, pad: float = 0.03, aspect: bool = True):
    x0, y0 = xy[:, 0].min(), xy[:, 1].min()
    x1, y1 = xy[:, 0].max(), xy[:, 1].max()
    dx, dy = (x1 - x0) * pad, (y1 - y0) * pad
    ax.set_xlim(x0 - dx, x1 + dx)
    ax.set_ylim(y0 - dy, y1 + dy)
    if aspect:
        ax.set_aspect("equal")


def page_mm(fig) -> tuple[float, float]:
    """Size of the PDF that ``fig`` would produce, in millimetres."""
    buf = io.BytesIO()
    fig.savefig(buf, format="pdf", facecolor="white")
    m = re.search(rb"/MediaBox\s*\[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.-]+)",
                  buf.getvalue())
    if not m:
        return (0.0, 0.0)
    x0, y0, x1, y1 = (float(v) for v in m.groups())
    return ((x1 - x0) / 72 * 25.4, (y1 - y0) / 72 * 25.4)


MIN_PT = 8.0      # nothing in a panel is set below this, journal floor is 8


def floor_text(fig, min_pt: float = MIN_PT) -> None:
    """Raise every text artist to at least ``min_pt``.

    Dense panels had annotations down to 5.2 pt - scale bars, corner notes,
    legend titles - which is unreadable at 158 mm and is the first thing a copy
    editor flags. The journal asks for 8 pt; 7 pt is the compromise here,
    because the alternative in a six-panel plate is an 8 pt label sitting on top
    of the imagery it labels.
    """
    from matplotlib.text import Text
    for t in fig.findobj(Text):
        if t.get_text() and t.get_fontsize() < min_pt:
            t.set_fontsize(min_pt)


def fit(fig, maxw: float = MAXW_MM, maxh: float = MAXH_MM) -> None:
    """Scale the canvas until the tight bounding box is exactly ``maxw`` wide.

    ``bbox_inches="tight"`` makes the exported page whatever the content needs,
    not what the canvas says, so a 158 mm canvas was exporting anywhere from 132
    to 193 mm. Two consequences, both fixed here: artwork wider than the text
    block gets scaled down by Word and 8 pt lettering prints at 6.6 pt, and
    artwork narrower leaves the figures in one paper at visibly different
    widths. The target is therefore met from both sides.

    Lettering is sized in points and does not scale with the canvas, so the step
    is damped and iterated rather than solved in one go.
    """
    for _ in range(10):
        w, h = page_mm(fig)
        if w <= 0:
            return
        k = maxw / w
        if h * k > maxh:                 # very tall figures: height is the limit
            k = maxh / h
        if abs(k - 1.0) < 0.004:
            return
        f = 1.0 + 0.85 * (k - 1.0)
        cw, ch = fig.get_size_inches()
        fig.set_size_inches(cw * f, ch * f)


def save(fig, outdir, name: str, formats=("pdf", "png"), dpi=600):
    """Write vector + raster next to each other; returns the paths.

    The canvas is fitted to the manuscript's text block first (see ``fit``), and
    the raster is written at 600 dpi: Springer asks 600 dpi for combination art
    and 1200 dpi for line art, and any lower value in this toolkit is rounded up
    rather than silently kept.
    """
    # A draw before the floor: tick labels are created lazily by the locators,
    # so findobj() before the first draw returns only some of them and a panel
    # that set labelsize=6.6 kept 6.6 pt while every other label was raised to 7.
    fig.canvas.draw()
    floor_text(fig)
    fit(fig)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in formats:
        p = outdir / f"{name}.{ext}"
        fig.savefig(p, format=ext, facecolor="white",
                    **({"dpi": max(dpi, 600)} if ext == "png" else {}))
        paths.append(p)
    plt.close(fig)
    return paths


def legend_clean(ax, **kw):
    kw.setdefault("handlelength", 1.4)
    kw.setdefault("handletextpad", 0.5)
    kw.setdefault("borderaxespad", 0.2)
    kw.setdefault("labelspacing", 0.28)
    leg = ax.legend(**kw)
    if leg is not None:
        leg.get_frame().set_linewidth(0)
    return leg


def hillshade_like(z: np.ndarray, dx: float, dy: float):
    """Light hillshade of a scalar field, used to give flat rasters some depth."""
    gy, gx = np.gradient(z, dy, dx)
    slope = np.pi / 2 - np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az, alt = np.radians(315.0), np.radians(45.0)
    shade = (np.sin(alt) * np.sin(slope) +
             np.cos(alt) * np.cos(slope) * np.cos(az - aspect))
    return np.clip(shade, 0, 1)
