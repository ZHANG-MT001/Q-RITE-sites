"""Case definitions and corridor statistics.

A "case" is a scenario: one study window and two plans (A straight, B with a
shifted mid-course). Everything a case reports - length, chains crossed, shafts
in the corridor, cavity length inside the corridor - is computed here, so the
figure and the table can never disagree.

The three cases are generated from the data, not hand-drawn: the fan's own
orientation decides what "perpendicular", "parallel" and "along the existing
road" mean for this particular field.
"""

from __future__ import annotations

import math

import numpy as np

from . import config, geo

HALF_WIDTH = 100.0          # corridor half width, m (+-100 m like the reference)
PLAN_LENGTH = 3000.0        # total span of a plan, m
SHIFT = 250.0               # lateral shift applied to the mid-course of plan B


def _chain_dirs(chains):
    """Unit direction of each chain axis."""
    out = []
    for c in chains:
        v = c[-1] - c[0]
        n = np.hypot(*v)
        if n > 1:
            out.append(v / n)
    return np.array(out)


def fan_orientation(chains) -> float:
    """Dominant chain bearing in radians, modulo pi (axes are undirected)."""
    dirs = _chain_dirs(chains)
    if len(dirs) == 0:
        return 0.0
    # double-angle average so opposite directions reinforce instead of cancel
    ang = np.arctan2(dirs[:, 1], dirs[:, 0])
    s = np.mean(np.sin(2 * ang))
    c = np.mean(np.cos(2 * ang))
    return 0.5 * np.arctan2(s, c)


def _clip_length(p, q, a, b, hw):
    """Length of segment pq inside the corridor rectangle of ab.

    Works in the corridor's own frame: s along the axis in [0, L], d across in
    [-hw, hw], clipped with the standard Liang-Barsky form (p*t <= q).
    """
    L = float(np.hypot(*(b - a)))
    if L < 1e-6:
        return 0.0
    u = (b - a) / L
    n = np.array([-u[1], u[0]])

    def to_sd(pt):
        r = pt - a
        return float(r @ u), float(r @ n)

    s1, d1 = to_sd(p)
    s2, d2 = to_sd(q)
    ds, dd = s2 - s1, d2 - d1

    t0, t1 = 0.0, 1.0
    for origin, delta, lo, hi in ((s1, ds, 0.0, L), (d1, dd, -hw, hw)):
        for pcoef, qcoef in ((-delta, origin - lo), (delta, hi - origin)):
            if abs(pcoef) < 1e-12:
                if qcoef < 0:
                    return 0.0
                continue
            r = qcoef / pcoef
            if pcoef < 0:
                t0 = max(t0, r)
            else:
                t1 = min(t1, r)
            if t0 > t1:
                return 0.0
    return float((t1 - t0) * math.hypot(ds, dd))


def _alpha(a, b, p, q):
    u = b - a
    v = q - p
    ang = abs(math.degrees(math.atan2(u[0] * v[1] - u[1] * v[0], u @ v)))
    ang = ang % 180.0
    return min(ang, 180.0 - ang)


def evaluate_plan(pts, chains, shafts, hw=HALF_WIDTH, states=None):
    """Corridor statistics for one plan (a polyline in metres).

    ``states`` is optional and only used to report the composition of the shafts
    the corridor touches. It is passed in rather than looked up so that callers
    which do not care keep the old signature.
    """
    pts = np.asarray(pts, float)
    length = float(sum(np.hypot(*(pts[i + 1] - pts[i])) for i in range(len(pts) - 1)))

    crossed, alphas, cavity = [], [], 0.0
    for ch in chains:
        hit = False
        for i in range(len(pts) - 1):
            for j in range(len(ch) - 1):
                if geo.polyline_crossings(np.vstack([pts[i], pts[i + 1]]),
                                          np.vstack([ch[j], ch[j + 1]])):
                    hit = True
                    alphas.append(_alpha(pts[i], pts[i + 1], ch[j], ch[j + 1]))
        if hit:
            crossed.append(ch)
        # length of this chain inside any plan segment's corridor
        clipped = 0.0
        for i in range(len(pts) - 1):
            for j in range(len(ch) - 1):
                clipped += _clip_length(ch[j], ch[j + 1], pts[i], pts[i + 1], hw)
        cavity += min(clipped, float(geo.polyline_length(ch)))

    # shafts inside the corridor
    in_corridor = 0
    if shafts is not None and len(shafts):
        d = np.full(len(shafts), 1e18)
        for i in range(len(pts) - 1):
            d = np.minimum(d, geo.point_segment_distance(
                shafts[:, 0], shafts[:, 1], pts[i, 0], pts[i, 1], pts[i + 1, 0], pts[i + 1, 1]))
        in_corridor = int((d <= hw).sum())
    n_active = n_inactive = None
    if states is not None and shafts is not None and len(shafts):
        m = d <= hw
        st = np.asarray(states)
        n_active = int((st[m] == "active").sum())
        n_inactive = int((st[m] == "inactive").sum())

    alphas = np.array(alphas) if alphas else np.array([])
    return {
        "length_km": round(length / 1000, 2),
        "lines": len(crossed),
        "shafts": in_corridor,
        "cavity_km": round(cavity / 1000, 3),
        "alpha_median": round(float(np.median(alphas)), 1) if len(alphas) else None,
        "n_orth": int((alphas >= 70).sum()),
        "n_obl": int(((alphas >= 45) & (alphas < 70)).sum()),
        "n_par": int((alphas < 45).sum()),
        "shafts_active": n_active,
        "shafts_inactive": n_inactive,
    }


def _plan_pair(center, direction, chains, shafts):
    """Straight plan A and a mid-course shifted plan B, in metres."""
    u = np.array([math.cos(direction), math.sin(direction)])
    n = np.array([-u[1], u[0]])
    a = center - u * PLAN_LENGTH / 2
    b = center + u * PLAN_LENGTH / 2

    best = None
    for sign in (+1, -1):
        mid = center + n * SHIFT * sign
        pts = np.vstack([a, mid, b])
        stats = evaluate_plan(pts, chains, shafts)
        if best is None or stats["lines"] < best[0]["lines"]:
            best = (stats, pts, sign)

    stats_a = evaluate_plan(np.vstack([a, b]), chains, shafts)
    return (np.vstack([a, b]), stats_a), (best[1], best[0]), best[2]


def build(chains, shafts, roads=None, center=None):
    """Return the case list consumed by the web map."""
    P = np.vstack([c for c in chains])
    center = np.array([np.median(P[:, 0]), np.median(P[:, 1])]) if center is None else np.asarray(center, float)
    fan = fan_orientation(chains)

    cases = []

    perp = fan + math.pi / 2
    (pa, sa), (pb, sb), sign = _plan_pair(center, perp, chains, shafts)
    cases.append({
        "id": "case1", "tab": "案例 1 · 正交穿越扇心",
        "title": "案例 1 — 横穿井链扇",
        "desc": "从扇心正交穿过。方案 A 直行，方案 B 中段横移 250 m。"
                "横移后穿越条数下降，但长度增加——这正是需要权衡的地方。",
        "plans": [("A 直行", pa, sa, "#A32D2D"), ("B 横移（推荐）", pb, sb, "#0F6E56")],
        "recommend": "recommended B",
        "center": center, "direction": perp,
    })

    par = fan
    (pa2, sa2), (pb2, sb2), _ = _plan_pair(center, par, chains, shafts)
    cases.append({
        "id": "case2", "tab": "案例 2 · 沿扇轴近平行",
        "title": "案例 2 — 与井链近平行",
        "desc": "沿井链走向布线。穿越条数少，但沿线始终贴着空腔，"
                "空洞长度 Lq 反而更大——近平行的隐蔽性就在这里。",
        "plans": [("A 平行", pa2, sa2, "#A32D2D"), ("B 横移（推荐）", pb2, sb2, "#0F6E56")],
        "recommend": "recommended B",
        "center": center, "direction": par,
    })
    return cases
