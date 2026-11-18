"""Second detection pass inside the corridor of a traced row.

The manual check said "the red line does not cover the whole row" about ten of 23
sampled rows. The line covers every shaft it accepted; the accepted set stops where
the first pass (conf 0.25) stopped, and the shortfall is concentrated at the row
ends. A global threshold cannot be lowered — it would buy false positives
everywhere — but the corridor of a row a human already confirmed is a different
problem: inside 12 m of that line the prior is high, so the threshold can drop.

Measured on Tabas Q01, same four windows, same framing, deduplicated at 6 m:
66 shafts at 0.25, 86 at 0.10. Every one of the 8 recovered points was a pit with
the same dark centre and pale ring as the accepted ones, sitting in a gap between
two accepted shafts. So the pass is worth running; the question this module has to
answer is whether it stays that clean on rows that run past a road or a spoil heap.

Two design points that are not obvious:

  * windows are sampled along the *extended* line, 150 m beyond each end. A corridor
    measured from the current polyline can never recover a shaft past the last
    accepted one, which is exactly where the losses are.
  * the extension is then grown back one shaft at a time, and stops at a gap wider
    than 3.5 times the row's own spacing. Without that rule a 150 m reach would join
    a row to whatever lies beyond it and the repair would be worse than the fault.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from . import detect, geo, iran_chains as ic, rows as rw

W_DEFAULT = "data/weights/best.pt"
CONF = 0.10
CORRIDOR = 12.0
EXTEND = 150.0
WIN_KM = 0.45
STEP_M = 380.0
DEDUPE = 6.0
GAP_FACTOR = 3.5
OUT = "data/ir/refine"


def _window(lat, lon, km=WIN_KM):
    dla = km / 2 / 111.32
    dlo = km / 2 / (111.32 * math.cos(math.radians(lat)))
    return (lon - dlo, lat - dla, lon + dlo, lat + dla)


def _curve(Q, f, step=30.0):
    """The row's own curve, sampled densely, for the corridor test.

    Using the window centres as the polyline would be wrong: they are 380 m apart,
    and on a bowed row the chord between two centres is metres away from the shafts
    it passes. Sampled every 30 m the polyline is the row.
    """
    lo, hi = -f["length"] / 2 - EXTEND, f["length"] / 2 + EXTEND
    xs = np.arange(lo, hi + step, step)
    rr = np.polyval(f["coef"], xs)
    return (f["c"][None, :] + xs[:, None] * f["u"][None, :]
            + rr[:, None] * f["v"][None, :])


def _centres(Q, f):
    """Window centres along the row, extended EXTEND beyond each end."""
    lo, hi = -f["length"] / 2 - EXTEND, f["length"] / 2 + EXTEND
    xs = list(np.arange(lo, hi + 1e-6, STEP_M))
    if len(xs) < 1:
        xs = [0.0]
    out = []
    for s in xs:
        rr = float(np.polyval(f["coef"], s))
        p = f["c"] + f["u"] * s + f["v"] * rr
        out.append(p)
    return np.array(out)


def collect(lab, cid, Q, f, zone, conf=CONF, weights=W_DEFAULT, cache=OUT):
    """One row: window centres, detections, corridor filter. Cached per row."""
    p = Path(cache) / f"{cid}.json"
    if p.exists():
        d = json.load(open(p, encoding="utf-8"))
        return d
    centres = _centres(Q, f)
    lon, lat = geo.to_lonlat(centres[:, 0], centres[:, 1], zone=zone)
    t0 = time.time()
    pts = []
    for a, b in zip(np.atleast_1d(lon), np.atleast_1d(lat)):
        pr, _ = detect.run(_window(b, a), zoom=19, weights=weights, conf=conf,
                           imgsz=640, cache="data/tiles_cache/ir", workers=8,
                           log=lambda *x, **kk: None)
        pts += pr
    # corridor test against the row's own curve, then dedupe
    E, N = geo.to_utm(np.array([q["lon"] for q in pts] or [0.0]),
                      np.array([q["lat"] for q in pts] or [0.0]), zone=zone)
    Pt = np.column_stack([E, N])
    keep = np.zeros(len(Pt), bool)
    if len(Pt):
        line = _curve(Q, f)
        dist, _ = ic.pt_seg(Pt, line[:-1], line[1:])
        keep = dist <= CORRIDOR
    Pt, cf = Pt[keep], np.array([q["conf"] for q in pts])[keep] if len(pts) else np.array([])
    ll = np.flatnonzero(keep)
    order = np.argsort(-cf) if len(cf) else []
    taken = np.zeros(len(Pt), bool)
    sel = []
    for i in order:
        if taken[i]:
            continue
        sel.append(int(ll[i]))
        taken |= np.hypot(Pt[:, 0] - Pt[i, 0], Pt[:, 1] - Pt[i, 1]) <= DEDUPE
    d = {"lab": lab, "cid": cid, "conf": conf, "zone": int(zone),
         "n_windows": len(centres), "n_raw": len(pts), "n_corridor": len(sel),
         "sec": round(time.time() - t0),
         "pts": [[round(float(pts[i]["lon"]), 6), round(float(pts[i]["lat"]), 6),
                  round(float(pts[i]["conf"]), 3)] for i in sel]}
    Path(cache).mkdir(parents=True, exist_ok=True)
    json.dump(d, open(p, "w"), ensure_ascii=False)
    return d


def load(lab, cache=OUT):
    """All cached corridor detections for a region, keyed by cid."""
    p = Path(cache)
    out = {}
    if not p.exists():
        return out
    for f in p.glob(f"{lab}-Q*.json"):
        d = json.load(open(f, encoding="utf-8"))
        out[d["cid"]] = d["pts"]
    return out


def merge(Q_core, extra_ll, zone):
    """Grow the core outward through the corridor points, then refit.

    Points between the first and last core shaft are accepted as they are — they
    sit in gaps that the first pass missed, which is the whole point. Points beyond
    either end are accepted one at a time while the step from the growing end stays
    within GAP_FACTOR times the core's own median spacing.

    Both paths pass a spacing gate, added after Bam S Q12 went from 13 shafts to 26
    against a manual count of 14: the corridor there follows a wash, and a lower
    threshold happily fires on whatever sits in it. A new shaft has to land within
    0.35–1.8 times the row's own spacing from its nearest accepted neighbour —
    tighter than that is a double detection or a clutter blob, wider is a different
    row or an outlier. Without the gate the repair is worse than the fault.
    """
    if not extra_ll:
        return Q_core, rw.fit_row(Q_core, np.arange(len(Q_core)))
    Ex = np.array(extra_ll, float)
    E, N = geo.to_utm(Ex[:, 0], Ex[:, 1], zone=zone)
    Pe = np.column_stack([E, N])
    # drop corridor points that coincide with an accepted shaft
    dmin = np.full(len(Pe), np.inf)
    for k in range(len(Q_core)):
        dmin = np.minimum(dmin, np.hypot(Pe[:, 0] - Q_core[k, 0], Pe[:, 1] - Q_core[k, 1]))
    Pe = Pe[dmin > DEDUPE]

    f0 = rw.fit_row(Q_core, np.arange(len(Q_core)))
    u, c = f0["u"], f0["c"]
    s_core = (Q_core - c) @ u
    s_new = (Pe - c) @ u
    r_new = (Pe - c) @ np.array([-u[1], u[0]])
    ok = np.abs(r_new) <= CORRIDOR
    Pe, s_new = Pe[ok], s_new[ok]

    s0, s1 = float(s_core.min()), float(s_core.max())
    sp = max(f0["spacing"], 8.0)
    lo_sp, hi_sp = 0.35 * sp, 1.8 * sp
    acc = [Q_core[k] for k in range(len(Q_core))]

    def fits(p):
        d = min(float(np.hypot(p[0] - a[0], p[1] - a[1])) for a in acc)
        return lo_sp <= d <= hi_sp

    inside = np.nonzero((s_new > s0) & (s_new < s1))[0]
    inside = inside[np.argsort(s_new[inside])]
    for i in inside:
        if fits(Pe[i]):
            acc.append(Pe[i])
    for direction in (1, -1):
        edge = s1 if direction > 0 else s0
        idx = np.nonzero((s_new - edge) * direction > 0)[0]
        idx = idx[np.argsort((s_new[idx] - edge) * direction)]
        for i in idx:
            if abs(s_new[i] - edge) > GAP_FACTOR * sp:
                break
            if abs(s_new[i] - edge) > EXTEND + sp:
                break
            if fits(Pe[i]):
                acc.append(Pe[i])
                edge = s_new[i]
    Q = np.vstack(acc) if len(acc) > len(Q_core) else Q_core
    f = rw.fit_row(Q, np.arange(len(Q)))
    # hand the points back in along-axis order: the caller measures road crossings
    # by joining consecutive points, so an unordered array turns a row into a
    # zigzag through itself and counts crossings that are not there
    Q = Q[np.asarray(f["order"], int)]
    return Q, f


def run(labs=("R3", "R4", "R6", "R7", "R8", "R11"), conf=CONF,
        weights=W_DEFAULT, cache=OUT):
    tot, t0 = 0, time.time()
    todo = []
    for lab in labs:
        kept, _ = ic.analyse(lab, f"data/ir/blocks11/block_{lab}.json", refine=False)
        d = json.load(open(f"data/ir/blocks11/block_{lab}.json", encoding="utf-8"))
        P, zone = ic.points_utm(d["pts"])
        by = {r["cid"]: r for r in kept}
        stats = {}
        found = rw.extract(P, stats=stats)
        # cids were assigned on the same order in analyse, so zip is safe
        for i, f in enumerate(found, 1):
            cid = f"{lab}-Q{i:02d}"
            if cid not in by:
                continue
            if (Path(cache) / f"{cid}.json").exists():
                continue
            Q = P[np.asarray(f["path"], int)]
            f2 = rw.fit_row(Q, np.arange(len(Q)))
            todo.append((lab, cid, Q, f2, zone))
    print(f"{len(todo)} rows to run at conf {conf}", flush=True)
    for k, (lab, cid, Q, f2, zone) in enumerate(todo, 1):
        d = collect(lab, cid, Q, f2, zone, conf=conf, weights=weights, cache=cache)
        tot += d["sec"]
        print(f"[{k}/{len(todo)}] {cid}  {d['n_windows']} win  {d['n_raw']} raw -> "
              f"{d['n_corridor']} in corridor  {d['sec']}s  (total {tot/60:.1f} min)",
              flush=True)
    print(f"done in {(time.time()-t0)/60:.1f} min")
    return tot


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    run(args or ("R3", "R4", "R6", "R7", "R8", "R11"))
