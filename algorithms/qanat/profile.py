"""Along-line interference profile and the plan sweep behind the core figure.

Two products, both computed from the same kernel:

``bins``   one plan -> interference intensity I(s) sampled every ``step`` metres.
           I(s) is the status-weighted cavity density: how many metres of cavity
           underlie each metre of alignment at chainage s, reduced by the state
           weight w. Dimensionless, and it varies along the line - which is the
           whole point, since a corridor total cannot say *where* the line is bad.

``scan``   many plans (bearing x lateral offset) -> one row per plan, so the
           crossing count, the shaft count and the cavity length can be plotted
           against the crossing angle. A corridor count is a single number per
           plan; only a sweep shows whether that number depends on geometry at
           all.

All geometry is in metres on the projected plane (UTM 45N). No reprojection here.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import cases, config, geo

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

INK = "#2C2C2A"
MUTED = "#6B6862"
WARM = "#BA7517"
RED = "#A32D2D"
AMBER = "#EF9F27"
GREEN = "#0F6E56"


# --- plan construction ---------------------------------------------------
def plan_from(center, bearing_deg: float, length: float, offset: float = 0.0):
    """Straight plan of ``length`` metres through ``center``, shifted sideways.

    bearing is a compass-style angle in degrees on the projected plane; the
    lateral offset is applied perpendicular to it.
    """
    th = math.radians(bearing_deg)
    u = np.array([math.cos(th), math.sin(th)])
    n = np.array([-u[1], u[0]])
    c = np.asarray(center, float) + n * offset
    a = c - u * length / 2
    b = c + u * length / 2
    return np.vstack([a, b])


def _sd(pts, a, u, n):
    r = np.asarray(pts, float) - a
    return r @ u, r @ n


def _clip_sd(s0, d0, s1, d1, s_lo, s_hi, hw):
    """Liang-Barsky in the corridor's own (s, d) frame. Returns (t0, t1)."""
    ds, dd = s1 - s0, d1 - d0
    t0, t1 = 0.0, 1.0
    for origin, delta, lo, hi in ((s0, ds, s_lo, s_hi), (d0, dd, -hw, hw)):
        for pcoef, qcoef in ((-delta, origin - lo), (delta, hi - origin)):
            if abs(pcoef) < 1e-12:
                if qcoef < 0:
                    return 1.0, 0.0
                continue
            r = qcoef / pcoef
            if pcoef < 0:
                t0 = max(t0, r)
            else:
                t1 = min(t1, r)
            if t0 > t1:
                return 1.0, 0.0
    return t0, t1


def bins(pts, chains, shafts, step=None, hw=cases.HALF_WIDTH, weights=None):
    """Interference profile of one plan.

    Returns a list of dicts, one per ``step``-metre bin along the alignment.
    """
    step = config.SEGMENT_LEN_M if step is None else float(step)
    pts = np.asarray(pts, float)
    length = float(geo.polyline_length(pts))
    nb = max(1, int(math.ceil(length / step)))

    cav = np.zeros(nb)          # metres of cavity inside the corridor, per bin
    n_cross = np.zeros(nb, int)
    alphas = [[] for _ in range(nb)]
    n_shaft = np.zeros(nb, int)
    w_shaft = np.zeros(nb)

    base = 0.0
    for i in range(len(pts) - 1):
        a = pts[i]
        seg = pts[i + 1] - a
        seglen = float(np.hypot(*seg))
        if seglen < 1e-9:
            continue
        u = seg / seglen
        n = np.array([-u[1], u[0]])

        for ch in chains:
            s, d = _sd(ch, a, u, n)
            for j in range(len(ch) - 1):
                s0, d0 = s[j], d[j]
                s1, d1 = s[j + 1], d[j + 1]
                Lseg = float(math.hypot(s1 - s0, d1 - d0))
                if Lseg < 1e-9:
                    continue
                t0, t1 = _clip_sd(s0, d0, s1, d1, 0.0, seglen, hw)
                if t1 <= t0:
                    continue
                clen = (t1 - t0) * Lseg

                if d0 * d1 < 0:  # this chain segment straddles the alignment
                    f = -d0 / (d1 - d0)
                    sc = s0 + (s1 - s0) * f
                    k = int((base + sc) // step)
                    if 0 <= k < nb:
                        n_cross[k] += 1
                        v = np.array([s1 - s0, d1 - d0])
                        ang = abs(math.degrees(math.atan2(u[0] * v[1] - u[1] * v[0], u @ v)))
                        ang %= 180.0
                        alphas[k].append(min(ang, 180.0 - ang))

                # Distribute the clipped length over the chainage interval it
                # actually occupies, in absolute chainage. Two traps here: the
                # interval must be sorted (chains whose vertices run against the
                # plan direction give S0 > S1, and an unsorted pair silently
                # contributes nothing), and the offset must be the cumulative
                # segment start, not i * step.
                S0 = base + s0 + (s1 - s0) * t0
                S1 = base + s0 + (s1 - s0) * t1
                if S0 > S1:
                    S0, S1 = S1, S0
                if S1 - S0 < 1e-9:
                    k = int(S0 // step)
                    if 0 <= k < nb:
                        cav[k] += clen
                    continue
                span = S1 - S0
                for k in range(max(0, int(S0 // step)), min(nb - 1, int(S1 // step)) + 1):
                    lo = max(S0, k * step)
                    hi = min(S1, (k + 1) * step)
                    if hi > lo:
                        cav[k] += clen * (hi - lo) / span

        base += seglen

    if shafts is not None and len(shafts):
        w = np.ones(len(shafts)) if weights is None else np.asarray(weights, float)
        # Distance to the polyline with clamped projection, i.e. exactly the
        # definition cases.evaluate_plan uses. Anything else and the profile
        # total disagrees with the corridor table at the two ends.
        best = np.full(len(shafts), 1e18)
        best_s = np.zeros(len(shafts))
        base = 0.0
        for i in range(len(pts) - 1):
            a = pts[i]
            seg = pts[i + 1] - a
            seglen = float(np.hypot(*seg))
            if seglen < 1e-9:
                continue
            u = seg / seglen
            n = np.array([-u[1], u[0]])
            s, d = _sd(shafts, a, u, n)
            dc = np.where(s < 0, np.hypot(s, d),
                          np.where(s > seglen, np.hypot(s - seglen, d), np.abs(d)))
            take = dc < best
            best = np.where(take, dc, best)
            best_s = np.where(take, base + np.clip(s, 0.0, seglen), best_s)
            base += seglen
        m = best <= hw
        for k, wk in zip((best_s[m] // step).astype(int), w[m]):
            if 0 <= k < nb:
                n_shaft[k] += 1
                w_shaft[k] += wk

    rows = []
    for k in range(nb):
        wb = (w_shaft[k] / n_shaft[k]) if n_shaft[k] else float(np.mean(weights) if weights is not None else 1.0)
        rows.append({
            "bin": k,
            "s_m": round(k * step + step / 2, 1),
            "n_cross": int(n_cross[k]),
            "alpha_deg": round(float(np.median(alphas[k])), 1) if alphas[k] else None,
            "cavity_m": round(float(cav[k]), 1),
            "cavity_ratio": round(float(cav[k] / step), 3),
            "shafts": int(n_shaft[k]),
            "w_bar": round(float(wb), 3),
            "intensity": round(float(cav[k] / step * wb), 3),
        })
    return rows, length


# --- plan sweep ----------------------------------------------------------
def scan(chains, shafts, center, bearings, offsets, length=cases.PLAN_LENGTH,
         hw=cases.HALF_WIDTH, fan_deg: float | None = None):
    """One row of corridor statistics per (bearing, offset) plan.

    ``rel_deg`` is the design variable: the angle between the plan and the fan
    axis, folded into 0-90. It is what a designer actually chooses, and unlike
    the median crossing angle it is not conditioned on a crossing happening -
    a near-parallel plan overlaps chains without crossing them, so the median
    crossing angle of such a plan is biased upwards and unusable as an axis.
    """
    rows = []
    for brg in bearings:
        for off in offsets:
            pts = plan_from(center, brg, length, off)
            st = cases.evaluate_plan(pts, chains, shafts, hw=hw)
            L = max(st["length_km"], 1e-6)
            rel = None
            if fan_deg is not None:
                d = abs((float(brg) - fan_deg)) % 180.0
                rel = round(min(d, 180.0 - d), 1)
            rows.append({
                # The raw bearing, not bearing % 180: a plan at 180 deg with
                # offset +x is a different line from one at 0 deg with +x (the
                # lateral offset flips with the direction), yet both fold to 0.
                # Folding here collapsed the 0/180 pair into nine duplicate keys
                # per site, and the duplicates disagreed - 10 crossings against
                # 23 at the same nominal cell. rel_deg folds separately below,
                # which is where folding belongs.
                "bearing_deg": round(float(brg), 1),
                "bearing_folded_deg": round(float(brg) % 180.0, 1),
                "rel_deg": rel,
                "offset_m": round(float(off), 1),
                "length_km": st["length_km"],
                "lines": st["lines"],
                "shafts": st["shafts"],
                "cavity_km": st["cavity_km"],
                "alpha_median": st["alpha_median"],
                "n_orth": st["n_orth"],
                "n_obl": st["n_obl"],
                "n_par": st["n_par"],
                "lines_per_km": round(st["lines"] / L, 3),
                "shafts_per_km": round(st["shafts"] / L, 1),
                "lq_per_km": round(st["cavity_km"] / L, 4),
            })
    return rows


# --- output --------------------------------------------------------------
def write_csv(rows, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("", encoding="utf-8")
        return p
    with p.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return p


def plot_profile(rows, out, title: str = "", plan=None, chains=None, step=None):
    """I(s) along the alignment, banded by percentile, with crossings marked."""
    s = np.array([r["s_m"] for r in rows])
    I = np.array([r["intensity"] for r in rows])
    sh = np.array([r["shafts"] for r in rows])
    p33, p66 = np.percentile(I[I > 0], [33, 66]) if (I > 0).any() else (0.3, 0.6)
    step = (s[1] - s[0]) if len(s) > 1 else 50.0

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10.5, 6.6), dpi=200,
                                  gridspec_kw={"height_ratios": [2.1, 1], "hspace": 0.12})

    lvl = np.where(I >= p66, 2, np.where(I >= p33, 1, 0))
    for k, col in ((0, "#E1F5EE"), (1, "#FAEEDA"), (2, "#F7C1C1")):
        ax.fill_between(s, 0, I, where=(lvl == k), step="mid", color=col, linewidth=0)
    ax.step(s, I, where="mid", color=INK, linewidth=1.0)
    ax.axhline(p33, color=MUTED, linewidth=0.6, linestyle=":")
    ax.axhline(p66, color=MUTED, linewidth=0.6, linestyle=":")
    ax.text(s[-1], p33, f"P33 = {p33:.2f} ", fontsize=7.5, color=MUTED, va="bottom",
            ha="right", bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.2))
    ax.text(s[-1], p66, f"P66 = {p66:.2f} ", fontsize=7.5, color=MUTED, va="bottom",
            ha="right", bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.2))

    for r in rows:
        if r["n_cross"] and r["alpha_deg"] is not None:
            ax.plot(r["s_m"], I[r["bin"]], marker="v", markersize=4.0,
                    color=RED if r["alpha_deg"] >= 60 else WARM,
                    markeredgecolor="white", markeredgewidth=0.5, zorder=6)
    ax.set_ylabel("干扰强度 I(s)\n（加权空腔密度）", fontsize=10)
    ax.set_xlim(s[0] - step / 2, s[-1] + step / 2)
    ax.set_ylim(0, max(I.max() * 1.45, 0.1))
    ax.grid(True, color="#E4E0D8", linewidth=0.5)
    ax.set_title(title or "沿线干扰剖面", fontsize=12.5, color=INK, loc="left")
    ax.text(1.0, 1.02, "▽ 红线=正交穿越 α≥60°　橙线=斜交 α<60°", transform=ax.transAxes,
            ha="right", fontsize=8, color=MUTED)

    ax2.bar(s, sh, width=step * 0.8, color="#C9A227", alpha=0.75, linewidth=0)
    ax2.set_ylabel("走廊内井口数", fontsize=9.5)
    ax2.set_xlabel("里程 s (m)", fontsize=10)
    ax2.set_xlim(s[0] - step / 2, s[-1] + step / 2)
    ax2.grid(True, axis="y", color="#E4E0D8", linewidth=0.5)

    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


def plot_scan(rows, out, title: str = ""):
    """Four panels: each corridor metric against the plan-to-fan angle."""
    a = np.array([r["rel_deg"] if r["rel_deg"] is not None else np.nan for r in rows])
    lines = np.array([r["lines_per_km"] for r in rows])
    shafts = np.array([r["shafts_per_km"] for r in rows])
    lq = np.array([r["lq_per_km"] for r in rows])
    ok = ~np.isnan(a)

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 8.4), dpi=200)
    bins = [0, 15, 30, 45, 60, 75, 90.01]
    labels = ["0-15\n近平行", "15-30", "30-45", "45-60", "60-75", "75-90\n正交"]

    offs = np.array([r["offset_m"] for r in rows], float)

    def _r2(y, cols):
        X = np.column_stack([np.ones(len(y))] + cols)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        ss = ((y - X @ beta) ** 2).sum()
        tot = ((y - y.mean()) ** 2).sum()
        return 1.0 - ss / tot if tot > 0 else 0.0

    def panel(ax, y, ylabel, colour):
        ax.scatter(a[ok], y[ok], s=7, color=colour, alpha=0.40, linewidth=0)
        cx, med, lo, hi = [], [], [], []
        for i in range(len(bins) - 1):
            m = ok & (a >= bins[i]) & (a < bins[i + 1])
            if m.sum() < 3:
                continue
            cx.append((bins[i] + min(bins[i + 1], 90)) / 2)
            med.append(np.median(y[m]))
            lo.append(np.percentile(y[m], 25))
            hi.append(np.percentile(y[m], 75))
        med, lo, hi = np.array(med), np.array(lo), np.array(hi)
        ax.errorbar(cx, med, yerr=[med - lo, hi - med], fmt="o-", color=INK,
                    linewidth=1.8, markersize=5, capsize=3, zorder=6)
        ax.set_xlabel("线与井链走向夹角 β (°)", fontsize=9.5)
        ax.set_ylabel(ylabel, fontsize=9.5)
        ax.set_xlim(-3, 93)
        ax.set_ylim(bottom=0)
        ax.set_xticks([7.5, 22.5, 37.5, 52.5, 67.5, 82.5])
        ax.set_xticklabels(labels, fontsize=7.6)
        ax.grid(True, color="#E4E0D8", linewidth=0.5)
        ang = np.radians(a[ok])
        r_b = _r2(y[ok], [np.sin(ang)])
        r_o = _r2(y[ok], [offs[ok]])
        ax.text(0.03, 0.95, f"R²(β) = {r_b:.2f}    R²(位置) = {r_o:.2f}",
                transform=ax.transAxes, fontsize=8.2, color=INK, va="top",
                bbox=dict(fc="white", ec="#D9D5CC", lw=0.5, pad=3))

    panel(axes[0][0], lines, "穿越条数 / km", WARM)
    axes[0][0].set_title("① 穿越条数 —— 由夹角决定，不是危害量", fontsize=10.5, loc="left")
    panel(axes[0][1], shafts, "走廊内井口数 / km", GREEN)
    axes[0][1].set_title("② 井口暴露 —— 均值与夹角无关，走廊之间方向相反", fontsize=10.5, loc="left")
    panel(axes[1][0], lq, "压占空腔长度 Lq / km", RED)
    axes[1][0].set_title("③ 压占空腔长度 —— 均值无趋势，走廊之间符号翻转", fontsize=10.5, loc="left")

    ax = axes[1][1]
    sc = ax.scatter(lines[ok], shafts[ok], s=10, c=a[ok], cmap="RdYlBu_r",
                    vmin=0, vmax=90, alpha=0.75, linewidth=0)
    ax.set_xlabel("穿越条数 / km", fontsize=9.5)
    ax.set_ylabel("走廊内井口数 / km", fontsize=9.5)
    ax.set_title("④ 同一穿越数下的井口跨度", fontsize=10.5, loc="left")
    ax.grid(True, color="#E4E0D8", linewidth=0.5)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("β (°)", fontsize=8.5)
    cb.ax.tick_params(labelsize=7.5)

    rho = np.corrcoef(lines[ok], shafts[ok])[0, 1] if ok.sum() > 2 else float("nan")
    extra = ""
    if ok.sum() > 5:
        q = np.polyfit(lines[ok], shafts[ok], 1)
        resid = shafts[ok] - np.polyval(q, lines[ok])
        extra = f"　同等穿越数下井口残差 95% 区间 ±{1.96 * resid.std():.0f}/km"
    fig.suptitle(title + extra if title else
                 f"选线扫描：{ok.sum()} 条候选线　ρ(穿越, 井口) = {rho:.2f}{extra}",
                 fontsize=12.0, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.965))

    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p, rho
