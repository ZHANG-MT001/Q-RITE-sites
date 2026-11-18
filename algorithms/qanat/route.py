"""Least-cost alignment generation over a cavity-exposure cost field.

The cost field is the point. The scan showed that crossing count is explained by
the plan's angle (R2 = 0.48) but explains almost none of the shaft exposure
(R2 = 0.00), so generating alignments by minimising crossings would just be
minimising an angle artefact. Here the edge cost is

    c(x) = 1 + lam * I(x),      I = rho_A * b_eff * w

where rho_A is the weighted chain length per unit area (m/m2) after a Gaussian
smoothing, b_eff an effective cavity influence width, and w the state weight.
Integrating c along a path therefore returns

    cost = length + lam * (cavity length met along the way),

i.e. a length term plus an explicit price for sitting on cavity. Increasing lam
slides the solution along the length-exposure trade-off; that curve, not a single
alignment, is the deliverable.

Grid and search are pure numpy + heapq: no shapely, no GDAL, no scipy.
"""

from __future__ import annotations

import csv
import heapq
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
RED = "#A32D2D"
GREEN = "#0F6E56"


# --- cost field ----------------------------------------------------------
def _blur(a: np.ndarray, sigma_cells: float) -> np.ndarray:
    """Separable Gaussian blur, numpy only.

    Smoothing matters for more than looks: a raw raster of thin lines gives a
    cost field of near-zero cells with isolated spikes, and a grid search then
    squeezes through single-cell gaps that no road could follow.
    """
    if sigma_cells <= 0:
        return a
    r = max(1, int(round(3 * sigma_cells)))
    x = np.arange(-r, r + 1, dtype=float)
    k = np.exp(-0.5 * (x / sigma_cells) ** 2)
    k /= k.sum()
    out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 1, a)
    out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 0, out)
    return out


def build_density(chains, shafts, extent, cell=15.0, sigma=12.0, b_eff=3.0,
                  weights=None, shaft_w=0.25):
    """Return (I, x0, y0, cell, info); I is the exposure rate, per metre.

    ``I`` already carries b_eff, so integrating it along a path returns a length
    in metres - the same currency as the path's own length. That is what makes
    ``lam`` readable: it is the exchange rate between a metre of pavement and a
    metre of cavity exposure, nothing else.

    extent is (xmin, ymin, xmax, ymax) in metres.

    ``shaft_w`` is the metres of equivalent cavity credited to one shaft, in
    units of the cell size. It exists because a shaft is a point, and a point
    contributes nothing to a length integral; without the term a chain whose
    axis was fitted loosely still charges the full interpolated length, while a
    chain with sparse but confidently detected heads charges nothing in the gaps.
    The default 0.25 puts ~4 m of cavity at each head at a 15 m cell, which is
    the shaft collar plus apron rather than the 1 m bore.
    """
    x0, y0, x1, y1 = extent
    # Snap the grid to a multiple of the cell. Without this the result depends on
    # the *extent* rather than on the data: shifting the box by 7 m moves every
    # cell boundary, which changes which gaps between chains are open, and on this
    # case moved the reported cavity by 11% with the model untouched. Padding by a
    # whole number of cells was harmless, so the bug only shows up when a caller
    # pads by an arbitrary amount - the position check did, and disagreed with the
    # headline run at the same corridor. A fixed lattice makes the field a function
    # of the data alone, and makes any two runs comparable by construction.
    x0 = math.floor(x0 / cell) * cell
    y0 = math.floor(y0 / cell) * cell
    x1 = math.ceil(x1 / cell) * cell
    y1 = math.ceil(y1 / cell) * cell
    nx = max(2, int(round((x1 - x0) / cell)))
    ny = max(2, int(round((y1 - y0) / cell)))
    dens = np.zeros((ny, nx))          # metres of chain landing in each cell

    for ch in chains:
        seg = np.diff(ch, axis=0)
        for j in range(len(seg)):
            L = float(np.hypot(*seg[j]))
            n = max(2, int(L / (cell * 0.5)))
            t = np.linspace(0, 1, n)
            pts = ch[j] + np.outer(t, seg[j])
            ix = np.clip(((pts[:, 0] - x0) / cell).astype(int), 0, nx - 1)
            iy = np.clip(((pts[:, 1] - y0) / cell).astype(int), 0, ny - 1)
            np.add.at(dens, (iy, ix), L / n)

    # shaft positions sharpen the field where a chain was reconstructed loosely
    if shafts is not None and len(shafts):
        w = np.ones(len(shafts)) if weights is None else np.asarray(weights, float)
        ix = np.clip(((shafts[:, 0] - x0) / cell).astype(int), 0, nx - 1)
        iy = np.clip(((shafts[:, 1] - y0) / cell).astype(int), 0, ny - 1)
        np.add.at(dens, (iy, ix), w * cell * shaft_w)

    rho = dens / (cell * cell)                      # m of cavity per m2
    rho = _blur(rho, sigma / cell)
    I = rho * b_eff                                  # dimensionless per metre

    info = {"nx": nx, "ny": ny, "cavity_km": float(dens.sum() / 1000),
            "I_max": float(I.max()), "I_mean": float(I.mean())}
    return I, x0, y0, cell, info


def build_cost(chains, shafts, extent, cell=15.0, sigma=12.0, b_eff=3.0,
               lam=5.0, weights=None, shaft_w=0.25):
    """Return (cost, x0, y0, cell, info) for an explicit price ``lam``.

    For a price that comes from the corridor itself rather than from the caller,
    see ``auto_lambda``.
    """
    I, x0, y0, cell, info = build_density(chains, shafts, extent, cell=cell,
                                          sigma=sigma, b_eff=b_eff,
                                          weights=weights, shaft_w=shaft_w)
    cost = 1.0 + lam * I
    info = {**info, "lam": lam,
            "cost_min": float(cost.min()), "cost_max": float(cost.max())}
    return cost, x0, y0, cell, info


# --- pricing -------------------------------------------------------------
# The price lambda is not a property of the terrain; it is a decision. Fixing it
# by hand makes the same number mean different things in different corridors,
# which is what the position check exposed: at lam = 15 a corridor laid over the
# densest part of the fan improved by -32% while one 400 m away improved by +4%.
#
# Quoting the price as a ratio (alpha) at least gives it a reading that does not
# depend on the units of this site: alpha = 1 says the exposure met along the
# straight line is worth as much as the pavement. It does NOT make one price
# correct everywhere, and `pricing` measures why - the price at which a corridor
# switches to going round instead of through moves by a factor of three over
# 800 m of lateral shift. The honest deliverable is therefore a menu of regimes
# with their price windows, not a tuned lambda.
def _resample(path, step):
    """Return (points evenly spaced in arc length, total length)."""
    path = np.asarray(path, float)
    seg = np.diff(path, axis=0)
    L = np.hypot(seg[:, 0], seg[:, 1])
    total = float(L.sum())
    if total <= 0:
        return path[:1], 0.0
    cum = np.concatenate([[0.0], np.cumsum(L)])
    n = max(2, int(math.ceil(total / max(step, 1e-6))) + 1)
    t = np.linspace(0.0, total, n)
    return np.column_stack([np.interp(t, cum, path[:, 0]),
                            np.interp(t, cum, path[:, 1])]), total


def exposure_integral(I, x0, y0, cell, path):
    """Metres of equivalent cavity met along a path, i.e. the integral of I ds.

    Because I = b_eff * rho and rho is cavity length per unit area, the integral
    over a perpendicular crossing of one chain is b_eff - one crossing, priced in
    metres. Sample at half a cell so a 15 m grid cannot step over a thin chain.
    """
    ny, nx = I.shape
    pts, total = _resample(np.asarray(path, float), cell * 0.5)
    ix = np.clip(((pts[:, 0] - x0) / cell).astype(int), 0, nx - 1)
    iy = np.clip(((pts[:, 1] - y0) / cell).astype(int), 0, ny - 1)
    step = total / max(len(pts) - 1, 1)
    return float(I[iy, ix].sum() * step)


def auto_lambda(I, x0, y0, cell, start, goal, alpha=1.0):
    """Price the corridor by its own exposure: lam = alpha * L / integral(I ds).

    ``alpha`` is the ratio the engineer chooses, and it is dimensionless: alpha =
    1 says the exposure met along the straight line is worth as much as the
    pavement itself, alpha = 0.5 says half as much. Invert it and it reads as an
    exposure budget - alpha = 1 buys a detour of one straight-line exposure
    (25 crossings x 3 m = 75 m here) to reach zero.

    The implied per-crossing price is kappa = lam * b_eff metres, so the result
    is still reportable in the same units as a fixed price; nothing is hidden.
    """
    straight = np.vstack([np.asarray(start, float), np.asarray(goal, float)])
    L = float(geo.polyline_length(straight))
    E = exposure_integral(I, x0, y0, cell, straight)
    if E <= 1e-9:
        return None, L, E          # nothing to avoid: any price returns the line
    return float(alpha * L / E), L, E


# --- search --------------------------------------------------------------
def solve(cost, x0, y0, cell, start, goal):
    """8-connected Dijkstra with trapezoidal edge cost. Returns a path (N,2)."""
    ny, nx = cost.shape

    def to_ij(p):
        return (int(round((p[1] - y0) / cell)), int(round((p[0] - x0) / cell)))

    si, sj = to_ij(start)
    gi, gj = to_ij(goal)
    for (i, j) in ((si, sj), (gi, gj)):
        if not (0 <= i < ny and 0 <= j < nx):
            raise ValueError("start or goal falls outside the cost grid")

    flat = cost.ravel()
    dist = np.full(nx * ny, np.inf)
    prev = np.full(nx * ny, -1, dtype=np.int64)
    s = si * nx + sj
    g = gi * nx + gj
    dist[s] = 0.0
    pq = [(0.0, s)]
    steps = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
             (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)),
             (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2)))

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        if u == g:
            break
        ui, uj = divmod(u, nx)
        for di, dj, scale in steps:
            vi, vj = ui + di, uj + dj
            if not (0 <= vi < ny and 0 <= vj < nx):
                continue
            v = vi * nx + vj
            step_len = scale * cell
            nd = d + step_len * 0.5 * (flat[u] + flat[v])
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    if not np.isfinite(dist[g]):
        raise RuntimeError("no path found")

    path = [g]
    while path[-1] != s:
        path.append(int(prev[path[-1]]))
    path.reverse()
    xy = np.array([[x0 + (u % nx + 0.5) * cell, y0 + (u // nx + 0.5) * cell] for u in path])
    return xy, float(dist[g])


def smooth(xy, cost, x0, y0, cell, eps=1e-9):
    """String-pulling over the Dijkstra path, accepted only if the objective drops.

    An 8-connected shortest path is an octile path: between two points whose
    bearing is not a multiple of 45 degrees it is systematically longer than the
    straight line (up to sqrt(2); +7% on this start-end pair). Left in, that bias
    contaminates every length comparison, including the cost of avoidance, which
    is the quantity the whole method reports.

    A shortcut is taken only when its integrated cost is no larger than the
    original sub-path's, measured with the same trapezoidal rule the search
    minimised. Using "no worse than the worst cell" instead was enough to
    straighten the staircase but not to keep monotonicity in lambda: a longer,
    slightly cheaper shortcut could still raise the reported cavity.
    """
    ny, nx = cost.shape

    def cell_at(p):
        i = min(ny - 1, max(0, int((p[1] - y0) / cell)))
        j = min(nx - 1, max(0, int((p[0] - x0) / cell)))
        return float(cost[i, j])

    node = np.array([cell_at(p) for p in xy])
    seg = np.hypot(*(np.diff(xy, axis=0).T))
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (node[:-1] + node[1:]) * seg)])

    def shortcut_cost(a, b, ca, cb):
        d = b - a
        L = float(math.hypot(*d))
        if L < 1e-9:
            return 0.0
        n = max(2, int(L / (cell * 0.5)))
        t = np.linspace(0.0, 1.0, n)
        P = a + np.outer(t, d)
        ix = np.clip(((P[:, 0] - x0) / cell).astype(int), 0, nx - 1)
        iy = np.clip(((P[:, 1] - y0) / cell).astype(int), 0, ny - 1)
        c = cost[iy, ix]
        return float((0.5 * (c[:-1] + c[1:]) * (L / (n - 1))).sum())

    out = [xy[0]]
    i, n = 0, len(xy)
    while i < n - 1:
        chosen = i + 1
        for j in range(n - 1, i + 1, -1):
            sc = shortcut_cost(xy[i], xy[j], node[i], node[j])
            if sc <= (cum[j] - cum[i]) * (1.0 + eps) + 1e-9:
                chosen = j
                break
        out.append(xy[chosen])
        i = chosen
    return np.asarray(out)


def round_corners(xy, radius=250.0, max_deflect=70.0, arc_step=10.0):
    """Replace sharp vertices with circular arcs of at least ``radius``.

    A least-cost path on a grid is a polyline with hard corners, which no road
    can follow. Rounding is not cosmetic: it makes the reported length an
    achievable length, and it lets the alignment be checked against a minimum
    radius, which is the first thing a highway engineer looks at.

    Returns (rounded_xy, min_radius_m, n_violations). A vertex whose adjacent
    segments are too short to host the tangent length is left sharp and counted
    as a violation - shrinking the radius to make it fit would produce a number
    that looks compliant and is not.
    """
    xy = np.asarray(xy, float)
    if len(xy) < 3:
        return xy, float("inf"), 0
    out = [xy[0]]
    radii = []
    bad = 0
    for i in range(1, len(xy) - 1):
        p0, p1, p2 = xy[i - 1], xy[i], xy[i + 1]
        v1, v2 = p1 - p0, p2 - p1
        l1, l2 = float(np.hypot(*v1)), float(np.hypot(*v2))
        if l1 < 1e-6 or l2 < 1e-6:
            continue
        u1, u2 = v1 / l1, v2 / l2
        cross = u1[0] * u2[1] - u1[1] * u2[0]
        theta = math.acos(max(-1.0, min(1.0, float(u1 @ u2))))
        if theta < math.radians(2.0):
            out.append(p1)
            continue
        T = radius * math.tan(theta / 2.0)
        if theta > math.radians(max_deflect) or T > 0.45 * min(l1, l2):
            out.append(p1)
            bad += 1
            continue
        R = radius
        a = p1 - u1 * T
        b = p1 + u2 * T
        side = 1.0 if cross > 0 else -1.0
        centre = a + side * np.array([-u1[1], u1[0]]) * R
        ang_a = math.atan2(*(a - centre)[::-1])
        ang_b = math.atan2(*(b - centre)[::-1])
        d = ang_b - ang_a
        while d > math.pi:
            d -= 2 * math.pi
        while d < -math.pi:
            d += 2 * math.pi
        n = max(4, int(abs(R * d) / arc_step))
        for k in range(n + 1):
            t = ang_a + d * k / n
            out.append(centre + R * np.array([math.cos(t), math.sin(t)]))
        radii.append(R)
    out.append(xy[-1])
    return np.asarray(out), (min(radii) if radii else float("inf")), bad


def fit_alignment(xy, cost, x0, y0, cell, radius=250.0,
                  tols=(15.0, 25.0, 40.0, 60.0, 90.0)):
    """Smooth, simplify and round, loosening the tolerance until the radius fits.

    The simplification tolerance and the achievable curve radius are coupled: a
    polyline whose segments are shorter than the tangent length cannot host the
    arc, and the leftover sharp vertex is exactly the thing a highway reviewer
    checks first. Rather than report a violation, the tolerance is raised until
    every bend fits - the first tolerance that produces no sharp vertex wins, and
    if none does, the fewest violations is kept and reported.
    """
    xy = smooth(xy, cost, x0, y0, cell)
    best = None
    for tol in tols:
        simp = simplify(xy, tol=tol)
        rounded, rmin, bad = round_corners(simp, radius=radius)
        cand = (bad, geo.polyline_length(rounded), rounded, rmin, tol)
        if best is None or (cand[0], cand[1]) < (best[0], best[1]):
            best = cand
        if bad == 0:
            break
    return best[2], best[3], best[0]


def simplify(xy, tol=6.0):
    """Douglas-Peucker: a grid path has a vertex every cell, which is not an alignment."""
    if len(xy) < 3:
        return xy

    def rec(pts):
        if len(pts) < 3:
            return pts
        a, b = pts[0], pts[-1]
        ab = b - a
        L = np.hypot(*ab)
        if L < 1e-9:
            d = np.hypot(*(pts - a).T)
        else:
            d = np.abs(np.cross(ab, pts - a)) / L
        k = int(np.argmax(d))
        if d[k] <= tol:
            return np.vstack([a, b])
        left = rec(pts[:k + 1])
        right = rec(pts[k:])
        return np.vstack([left[:-1], right])

    return rec(xy)


# --- output --------------------------------------------------------------
def objective(xy, cost, x0, y0, cell):
    """Integrated cost of a finished alignment, with the search's own rule.

    The search minimises this quantity; the paper reports corridor statistics
    measured on the same polyline afterwards. They are different functionals of
    the same path - a blurred line integral versus a half-width clip - so a
    slightly larger lambda can land on a marginally larger reported cavity
    without the optimiser having failed. Reporting both, rather than only the
    one that looks monotone, is the honest way to present the trade-off.

    Each segment is resampled at half a cell. Sampling only the segment midpoint
    was wrong by up to two orders of magnitude: after simplification a segment
    can be hundreds of metres long, and one sample misses every cell it crosses,
    so the integral collapsed to almost exactly the path length.
    """
    ny, nx = cost.shape
    xy = np.asarray(xy, float)
    if len(xy) < 2:
        return 0.0
    total = 0.0
    for a, b in zip(xy[:-1], xy[1:]):
        d = b - a
        L = float(math.hypot(*d))
        if L < 1e-9:
            continue
        n = max(2, int(L / (cell * 0.5)))
        t = np.linspace(0.0, 1.0, n)
        P = a + np.outer(t, d)
        ix = np.clip(((P[:, 0] - x0) / cell).astype(int), 0, nx - 1)
        iy = np.clip(((P[:, 1] - y0) / cell).astype(int), 0, ny - 1)
        c = cost[iy, ix]
        total += float((0.5 * (c[:-1] + c[1:]) * (L / (n - 1))).sum())
    return total


def cavity_integral(xy, cost, x0, y0, cell, lam):
    """Metres of cavity-equivalent the path sits on: the (J - L) / lambda term.

    Independent of lambda, which is the point: the price decides how much that
    quantity is worth, not how much of it there is.
    """
    xy = np.asarray(xy, float)
    L = float(geo.polyline_length(xy))
    return (objective(xy, cost, x0, y0, cell) - L) / max(lam, 1e-9)


def metrics(pts, chains, shafts, hw=cases.HALF_WIDTH):
    st = cases.evaluate_plan(pts, chains, shafts, hw=hw)
    L = max(st["length_km"], 1e-9)
    st["lines_per_km"] = round(st["lines"] / L, 3)
    st["shafts_per_km"] = round(st["shafts"] / L, 1)
    st["lq_per_km"] = round(st["cavity_km"] / L, 4)
    return st


def write_row_csv(rows, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return p


def plot(cost, x0, y0, cell, paths, chains, out, title="", clean=None):
    """Cost surface with the straight plan, the optimised path and the clean one."""
    ny, nx = cost.shape
    ext = (x0 / 1000, (x0 + nx * cell) / 1000, y0 / 1000, (y0 + ny * cell) / 1000)
    fig, ax = plt.subplots(figsize=(9.6, 8.0), dpi=200)

    im = ax.imshow(cost, origin="lower", extent=ext, cmap="YlOrRd",
                   vmin=1.0, vmax=float(np.percentile(cost, 99.5)), alpha=0.92)
    cb = fig.colorbar(im, ax=ax, shrink=0.82)
    cb.set_label("通行代价 c = 1 + λ·I", fontsize=9)

    for ch in chains:
        ax.plot(ch[:, 0] / 1000, ch[:, 1] / 1000, color="#5F5E5A", linewidth=0.45,
                alpha=0.55, zorder=3)

    styles = {
        "straight": ("#2C2C2A", "--", 1.8, "直线方案（对照）"),
        "route":    (RED, "-", 2.6, "优化方案"),
        "clean":    (GREEN, "-", 2.6, "彻底避让（λ→∞）"),
    }
    seen = set()
    for key, (xy, st) in paths.items():
        col, ls, lw, label = styles[key]
        ax.plot(xy[:, 0] / 1000, xy[:, 1] / 1000, color=col, linestyle=ls,
                linewidth=lw, zorder=6, label=label if key not in seen else None)
        seen.add(key)
        ax.plot(xy[0, 0] / 1000, xy[0, 1] / 1000, marker="o", color=col, markersize=6, zorder=7)
        ax.plot(xy[-1, 0] / 1000, xy[-1, 1] / 1000, marker="s", color=col, markersize=6, zorder=7)

    ax.set_xlabel("E (km, UTM 45N)", fontsize=9.5)
    ax.set_ylabel("N (km, UTM 45N)", fontsize=9.5)
    ax.set_aspect("equal")
    ax.set_title(title or "选线生成：代价场与路径", fontsize=12.5, loc="left")
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.92)
    ax.tick_params(labelsize=8)

    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


def plot_tradeoff(rows, out, title=""):
    """Length gained vs cavity avoided, drawn as the Pareto envelope.

    The search minimises the integral of a blurred cost field, while the y axis
    is the corridor metric measured afterwards. The two are close but not
    identical, so a handful of points come out slightly dominated (a larger
    lambda landing on marginally more cavity). That is a resolution artefact
    plus a definition difference, not a modelling result - so the line follows
    the non-dominated envelope and the raw solves are kept as faint dots.
    """
    fig, ax = plt.subplots(figsize=(7.6, 5.4), dpi=200)
    L = np.array([r["length_km"] for r in rows])
    Q = np.array([r["cavity_km"] for r in rows])
    lam = np.array([r["lam"] for r in rows])

    order = np.argsort(L)
    front, best = [], np.inf
    for k in order:
        if Q[k] < best - 1e-9:
            front.append(k)
            best = Q[k]
    front = np.array(front)

    ax.scatter(L, Q, s=26, color="#D9D5CC", zorder=2, linewidth=0)
    ax.plot(L[front], Q[front], "-", color=MUTED, linewidth=1.3, zorder=3)
    sc = ax.scatter(L[front], Q[front], s=52, c=np.log10(np.maximum(lam[front], 1e-3)),
                    cmap="viridis", zorder=4, edgecolor="white", linewidth=0.9)
    for idx, k in enumerate(front):
        text = "λ→∞（彻底避让）" if lam[k] >= 1e5 else f"λ={lam[k]:g}"
        ax.annotate(text, (L[k], Q[k]), textcoords="offset points",
                    xytext=(8, 6) if idx % 2 == 0 else (8, -13),
                    fontsize=7.8, color=INK)
    # label the dominated solves too, faintly, so no dot is unexplained
    for k in range(len(L)):
        if k in set(front.tolist()):
            continue
        ax.annotate(f"λ={lam[k]:g}", (L[k], Q[k]), textcoords="offset points",
                    xytext=(6, -11), fontsize=6.8, color="#A8A49B")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("log10 λ", fontsize=8.5)
    cb.ax.tick_params(labelsize=7.5)
    ax.set_xlabel("线路长度 (km)", fontsize=10)
    ax.set_ylabel("走廊内压占空腔 Lq (km)", fontsize=10)
    ax.set_title(title or "长度—暴露权衡曲线（非支配包络）", fontsize=12, loc="left")
    ax.grid(True, color="#E4E0D8", linewidth=0.5)

    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p
