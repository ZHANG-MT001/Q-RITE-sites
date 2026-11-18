"""Trace shaft rows by following them, instead of clustering and fitting a chord.

Why this module exists
----------------------
The first reconstruction clustered detections by single linkage and then fitted a
straight line to each cluster. It failed in a specific and instructive way: at a
40 m link radius, rows that are 50-80 m apart get welded into one blob, and the
blob is too curved to pass a straightness test, so the *whole* row system was
dropped rather than split. In Bam S the three largest clusters (152, 125 and 52
points) had straight-line residuals of 75, 54 and 34 m; all three were discarded,
and the figure showed only the small fragments that happened to stay straight.
Widening the radius makes the welding worse, so the radius is the wrong knob.

What works is what a photo-interpreter does by eye: start on a shaft, look along
the row, take the next shaft in line, keep going. That walk has two properties
the cluster fit lacks — it follows a curved row without requiring the row to be
straight, and its lateral gate is a few metres, so it cannot step sideways onto
the neighbouring row.

Geometry
--------
`trace_all` returns ordered point indices. `fit_row` fits
    r = a*s^2 + b*s + c
in the row's own principal frame and reports the residual *after* the quadratic.
That is the number that means something here: a qanat row is an arc along a
contour, and a straight-line residual conflates curvature with noise.
"""
from __future__ import annotations

import math

import numpy as np

# Lateral gate for the walk. Rows in these fields are 50-80 m apart and the gate
# is an order of magnitude smaller: a walk that can reach the neighbouring row is
# worse than one that stops early, because the first error is invisible in the
# output and the second is a visible gap.
TOL_LAT = 7.0
# The gate opens very slowly with distance walked. Without it a row drifting 3
# degrees off the current heading loses its own shafts after ~130 m; with a
# larger coefficient the walk can cut across the inside of a bend.
LAT_GROW = 0.03
STEP_MIN = 4.0
STEP_MAX = 100.0      # a 100 m gap is a buried or ploughed stretch, not a new row
TURN_MAX = 25.0       # heading change allowed per step, degrees
SMOOTH = 0.55         # weight on the old heading when updating
DEDUPE = 0.6          # overlap above which two traces are the same row
SEED_RAD = 70.0
# Seeds must have a neighbour close enough to define a heading, or a lone detection
# on a slope would be walked into a row. The first cut set this to 60 m, which
# silently excluded the *sparsest* rows: in Tabas the shafts are 60-80 m apart, so a
# real row of ten shafts 560 m long was never seeded at all, and the block reported
# four rows where there are five. The ceiling is now 90 m — above the widest row
# spacing the detector is expected to meet, and still below the distance at which a
# detection has no row-mate at all.
SEED_MAX_NN = 90.0
SEED_GROW = 1.4       # seed neighbourhood radius, as a multiple of the nn distance

# A walk may bridge a gap — shafts are ploughed over, buried, or simply missed by
# the detector — but bridging is where a trace stops being evidence and starts
# being a guess. A 15 m row with a 90 m hole is not one row with a hole; it is two
# rows, and joining them draws a straight line across ground where nothing was
# detected. So a trace is cut wherever one step exceeds SPLIT_FACTOR times the
# trace's own median step, with SPLIT_FLOOR as an absolute floor.
SPLIT_FACTOR = 3.5
SPLIT_FLOOR = 45.0

MIN_N = 5
MIN_LEN_M = 70.0
MAX_RESID_M = 8.0
SPACING_RANGE = (8.0, 80.0)
MIN_SPACING_DUP = 4.0  # two detections this close are one shaft
# A qanat row is an arc along a contour: it can bend, but not double back. A walk
# that accumulates more turning than this has left its row and is stitching two
# rows together through the noise between them.
MAX_TURN_DEG = 70.0


def _unit(v):
    n = float(np.hypot(v[0], v[1]))
    return v / n if n > 1e-9 else np.array([1.0, 0.0])


def _seed_dir(P, i, rad=SEED_RAD):
    """Initial heading: principal axis of the seed's neighbourhood."""
    d = np.hypot(P[:, 0] - P[i, 0], P[:, 1] - P[i, 1])
    nb = P[d <= rad]
    if len(nb) < 3:
        return None
    X = nb - nb.mean(0)
    w, V = np.linalg.eigh(X.T @ X)
    if w.max() < 1.0:
        return None
    u = V[:, np.argmax(w)]
    return _unit(u)


def _walk(P, seed, d0, tol_lat=TOL_LAT, step_max=STEP_MAX, used=None):
    """Follow a row from one seed in both directions; returns ordered indices."""
    order = [seed]
    for sign in (1.0, -1.0):
        d = d0 * sign
        cur = P[seed]
        while True:
            rel = P - cur
            dist = np.hypot(rel[:, 0], rel[:, 1])
            along = rel @ d
            lat = np.abs(rel @ np.array([-d[1], d[0]]))
            gate = tol_lat + LAT_GROW * np.maximum(along, 0.0)
            ok = ((along > STEP_MIN) & (along < step_max) & (lat <= gate)
                  & (dist <= step_max * 1.45))
            if used is not None:
                ok = ok & (~used)
            if not ok.any():
                break
            j = int(np.nonzero(ok)[0][np.argmin(along[ok])])
            step = _unit(P[j] - cur)
            if math.degrees(math.acos(max(-1.0, min(1.0, float(step @ d))))) > TURN_MAX:
                break
            d = _unit(SMOOTH * d + (1.0 - SMOOTH) * step)
            if sign > 0:
                order.append(j)
            else:
                order.insert(0, j)
            if used is not None:
                used[j] = True
            cur = P[j]
    return order


def trace_all(P, tol_lat=TOL_LAT, step_max=STEP_MAX):
    """Trace every row, then drop traces that merely re-walk another row."""
    n = len(P)
    if n < MIN_N:
        return []
    # nearest neighbour, used to order seeds and to reject isolated points
    nn = np.full(n, np.inf)
    for k in range(n):
        dd = np.hypot(P[:, 0] - P[k, 0], P[:, 1] - P[k, 1])
        dd[k] = np.inf
        if n > 1:
            nn[k] = dd.min()
    traces = []
    for i in np.argsort(nn):
        if nn[i] > SEED_MAX_NN:
            break
        d0 = _seed_dir(P, int(i), rad=max(SEED_RAD, SEED_GROW * nn[i]))
        if d0 is None:
            continue
        tr = _walk(P, int(i), d0, tol_lat=tol_lat, step_max=step_max)
        if len(tr) >= MIN_N:
            traces.append(tr)

    traces.sort(key=len, reverse=True)
    keep = []
    for tr in traces:
        s = set(tr)
        if any(len(s & set(k)) / min(len(s), len(k)) > DEDUPE for k in keep):
            continue
        keep.append(tr)
    return keep


def fit_row(P, idx):
    """Quadratic fit in the row frame: length, curvature-corrected residual, ends."""
    idx = np.asarray(idx, int)
    Q = P[idx]
    c = Q.mean(0)
    X = Q - c
    w, V = np.linalg.eigh(X.T @ X)
    u = V[:, np.argmax(w)]
    if u[1] < 0:
        u = -u
    v = np.array([-u[1], u[0]])
    s = X @ u
    r = X @ v
    o = np.argsort(s)
    # Only s and r are reordered. Writing `s, r, o = s[o], r[o], o[o]` instead
    # composes the permutation with itself and hands back a scrambled point order:
    # every measurement that reads s is still right, but the path drawn from it
    # zigzags, and a 210 m jump appeared inside a 324 m row. The order has to stay
    # the sorted index itself.
    s, r = s[o], r[o]
    # two detections on one shaft are one shaft
    dedup = [0]
    for k in range(1, len(s)):
        if s[k] - s[dedup[-1]] >= MIN_SPACING_DUP:
            dedup.append(k)
    dedup = np.asarray(dedup, int)
    s, r = s[dedup], r[dedup]
    order = idx[o[dedup]].tolist()
    n = len(s)
    if n >= 3:
        A = np.vstack([s ** 2, s, np.ones_like(s)]).T
        coef, *_ = np.linalg.lstsq(A, r, rcond=None)
        rr = r - A @ coef
    else:
        coef = np.array([0.0, 0.0, float(np.mean(r))])
        rr = r - coef[2]
    gaps = np.diff(s)
    sp = float(np.median(gaps)) if len(gaps) else float("nan")
    sd = float(np.std(gaps)) if len(gaps) > 1 else float("nan")
    s0, s1 = float(s.min()), float(s.max())
    ends = []
    for ss in (s0, s1):
        q = c + u * ss + v * float(np.polyval(coef, ss))
        ends.append((float(q[0]), float(q[1])))
    # sagitta: how far the row bows away from its own chord, in metres
    chord = np.polyval(coef, s0) + (np.polyval(coef, s1) - np.polyval(coef, s0)) * \
        (s - s0) / max(s1 - s0, 1e-9)
    # total turning between the two ends, from the slope dr/ds of the fit
    t0 = math.atan(2 * coef[0] * s0 + coef[1])
    t1 = math.atan(2 * coef[0] * s1 + coef[1])
    turn = abs(math.degrees(t1 - t0))
    return dict(n=n, length=float(s1 - s0), resid=float(np.sqrt(np.mean(rr ** 2))),
                spacing=sp, sp_sd=sd, gap_max=float(gaps.max()) if len(gaps) else float("nan"),
                sag=float(np.max(np.abs(r - chord))), turn=turn,
                bearing=float(math.degrees(math.atan2(u[0], u[1])) % 180.0),
                coef=coef, c=c, u=u, v=v, ends=ends, order=order)


def curve_utm(fit, samples=9, dx=0.0):
    """The fitted curve, sampled, in metres-UTM (for KML and figures)."""
    s = np.linspace(-dx, fit["length"] + dx, samples)
    r = np.polyval(fit["coef"], s)
    return fit["c"][None, :] + s[:, None] * fit["u"][None, :] + r[:, None] * fit["v"][None, :]


def _segments(P, tr):
    """Cut a trace where it jumped a hole far wider than its own spacing."""
    if len(tr) < 3:
        return [list(tr)]
    D = np.hypot(np.diff(P[tr][:, 0]), np.diff(P[tr][:, 1]))
    cut = max(SPLIT_FLOOR, SPLIT_FACTOR * float(np.median(D)))
    segs, cur = [], [tr[0]]
    for i, dd in enumerate(D):
        if dd > cut:
            segs.append(cur)
            cur = [tr[i + 1]]
        else:
            cur.append(tr[i + 1])
    segs.append(cur)
    return [s for s in segs if len(s) >= 3]


def extract(P, tol_lat=TOL_LAT, step_max=STEP_MAX, min_n=MIN_N,
            min_len=MIN_LEN_M, max_resid=MAX_RESID_M, spacing=SPACING_RANGE,
            max_turn=MAX_TURN_DEG, stats=None):
    """Rows that survive the filters, largest first, using each detection once.

    A detection belongs to one shaft and a shaft to one row, so the traces are
    made disjoint before they are fitted: longest first, each keeping only the
    points no earlier trace claimed. Without this the same three shafts at the
    junction of two walking directions are counted twice and "purity" — the share
    of detections on a row — can exceed 100%, which is a number a reviewer is
    right not to trust.
    """
    segs = []
    for tr in trace_all(P, tol_lat=tol_lat, step_max=step_max):
        segs.extend(_segments(P, tr))
    segs.sort(key=len, reverse=True)

    claimed = np.zeros(len(P), bool)
    out = []
    for tr in segs:
        free = [i for i in tr if not claimed[i]]
        if len(free) < len(tr) * DEDUPE:      # mostly a re-walk of a claimed row
            continue
        if len(free) < min_n:
            continue
        f = fit_row(P, free)
        if f["n"] < min_n or f["length"] < min_len or f["resid"] > max_resid:
            continue
        if not (spacing[0] <= f["spacing"] <= spacing[1]):
            continue
        if f["turn"] > max_turn:
            continue
        claimed[free] = True
        f["path"] = [int(i) for i in free]
        out.append(f)
    out.sort(key=lambda r: -r["n"])
    if stats is not None:
        stats.update(n_segments=len(segs), n_rows=len(out),
                     n_claimed=int(claimed.sum()))
    return out
