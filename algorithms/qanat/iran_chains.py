"""Turn screening detections into chain candidates, for the Iran windows.

This is a screening reconstruction, not the Turpan pipeline. A Turpan chain is
fitted from a hand-verified shaft list and graded on residual, count and cover;
here the only evidence is a detector run on a 0.45 km grid, so the output is a
*list of places to look*, with the geometry that would make a chain plausible.

Two reconstructions have been used here and the second one replaced the first for
a reason worth keeping in the record.

v1 — single linkage at 40 m, then a straight line through each cluster. It fails
in a way that is easy to miss from the numbers alone: rows 50-80 m apart get
welded into one blob, the blob is too curved for the straightness test, and the
*whole* row system is discarded rather than split. In Bam S the three largest
clusters (152, 125, 52 points) had straight-line residuals of 75, 54 and 34 m and
all three were dropped, so a figure that claimed to show the reconstruction
showed only the small fragments that happened to stay straight — 24% of the
detections. Widening the radius makes the welding worse.

v2 — walk each row: start on a shaft, look along the current heading, take the
next shaft inside a 7 m lateral gate, update the heading, repeat (qanat/rows.py).
The gate is small enough that the walk cannot step onto the next row, and the
heading follows a curved row so the row never has to be straight. Coverage in the
same window goes to 71%, and the drawn chain reaches both ends of the row.

The lesson for the paper is not "use tracing". It is that purity measured on a
reconstruction that silently drops its hardest cases is not a measure of the
detector; the first number looked conservative and was in fact an artefact.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from . import geo
from . import rows as rw

# Kept for the record and for the sensitivity table: the link radius was the
# parameter everyone reached for first, and it is the wrong knob (see above).
LINK_M = 40.0
MIN_N = rw.MIN_N
MIN_LEN_M = rw.MIN_LEN_M
MAX_RESID_M = rw.MAX_RESID_M
SPACING_RANGE = rw.SPACING_RANGE

PAVED = ("motorway", "trunk", "primary", "secondary", "tertiary")
MAJOR = ("motorway", "trunk", "primary", "secondary")

# The corridor second pass is only applied to rows with this many shafts in the
# first pass, because the 23-row manual sample says that is where it is calibrated:
# against the co-author's own counts, rows with >=12 shafts came back at 95% of his
# number, rows with 8-11 at 119% and rows with 5-7 at 150%. A five-shaft row gives a
# poor spacing estimate and almost nothing for the spacing gate to anchor on, so the
# pass inflates exactly the rows where the geometry is already weakest.
REFINE_MIN_N = 12

COORD = {"lat", "lon", "lat0", "lon0", "lat1", "lon1"}


def load_row_verdicts(path="data/ir/row_verdicts.json"):
    """Rows the co-author rejected, keyed by cid.

    Applied here rather than in the reporting layer so every product — csv, kml,
    figure, table — drops the same rows. Filtering only in the table is how a
    rejected row survives into a figure and is noticed by a reviewer instead.
    """
    p = Path(path)
    if not p.exists():
        return {}
    d = json.load(open(p, encoding="utf-8"))
    return {k: v for k, v in d.items() if not k.startswith("_") and v.get("verdict") == "no"}


def _cell(k, v):
    """Round for the csv.

    Coordinates to 6 decimals, everything else to 2. Rounding the whole row to 2
    silently turned 36.43909 into 36.44 — a kilometre of error in a file whose
    only purpose is to be pasted into Google Earth.
    """
    if isinstance(v, float):
        return round(v, 6) if k in COORD else round(v, 2)
    return v


def zone_of(lon):
    """UTM zone for a longitude. config.UTM_ZONE is Turpan (45N); Iran is 38-41."""
    return int((float(lon) + 180.0) // 6.0) + 1


def points_utm(pts):
    """Detections -> UTM metres in the right zone, plus that zone.

    Using the Turpan zone here is not a harmless offset: at 32 degrees from the
    central meridian the round trip back to lon/lat is off by 8 km, which silently
    puts every drawn chain in the wrong place.
    """
    lons = np.array([p["lon"] for p in pts], float)
    lats = np.array([p["lat"] for p in pts], float)
    zone = zone_of(lons.mean())
    e, n = geo.to_utm(lons, lats, zone=zone)
    return np.column_stack([e, n]), zone


def _cluster(P, link=LINK_M):
    """Single-linkage clustering by union-find; returns a label per point."""
    n = len(P)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        a, b = find(i), find(j)
        if a != b:
            parent[a] = b

    D = np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1])
    for i in range(n):
        for j in np.nonzero(D[i] <= link)[0]:
            if j > i:
                union(i, j)
    lab = np.array([find(i) for i in range(n)])
    # relabel 0..k-1 in order of first appearance
    seen, out = {}, np.zeros(n, int)
    for i, l in enumerate(lab):
        if l not in seen:
            seen[l] = len(seen)
        out[i] = seen[l]
    return out


def fit_chain(P):
    """Principal-axis fit: returns along-axis length, residual, spacing, bearing."""
    c = P.mean(0)
    X = P - c
    # principal direction as the eigenvector of the largest eigenvalue
    w, V = np.linalg.eigh(X.T @ X)
    u = V[:, np.argmax(w)]
    if u[1] < 0:                      # keep a stable sign for the bearing
        u = -u
    s = X @ u
    r = X @ np.array([-u[1], u[0]])
    order = np.argsort(s)
    s_sorted = s[order]
    gaps = np.diff(s_sorted)
    spacing = float(np.median(gaps)) if len(gaps) else float("nan")
    bearing = float(math.degrees(math.atan2(u[0], u[1])) % 180.0)
    return dict(
        lat=float(c[0]) if False else None,   # filled by caller (lat/lon, not UTM)
        cx=float(c[0]), cy=float(c[1]),
        length=float(s.max() - s.min()),
        resid=float(np.sqrt(np.mean(r ** 2))),
        spacing=spacing,
        n=int(len(P)),
        bearing=bearing,
    )


def segments_for(lab, roads_dir="data/ir/roads11", classes=PAVED, zone=None):
    p = Path(roads_dir) / f"{lab}.json"
    if not p.exists():
        return np.zeros((0, 2)), np.zeros((0, 2)), []
    ways = json.load(open(p, encoding="utf-8"))
    A, B, meta = [], [], []
    for w in ways:
        if w["cls"] not in classes or len(w["pt"]) < 2:
            continue
        a = np.array(w["pt"], float)
        e, n = geo.to_utm(a[:, 1], a[:, 0], zone=zone)
        P = np.column_stack([e, n])
        for i in range(len(P) - 1):
            A.append(P[i]); B.append(P[i + 1])
            meta.append((w["cls"], w.get("ref", "")))
    if not A:
        return np.zeros((0, 2)), np.zeros((0, 2)), []
    return np.array(A), np.array(B), meta


def pt_seg(P, A, B):
    best = np.full(len(P), np.inf)
    lab = np.full(len(P), -1)
    for k in range(len(A)):
        a, b = A[k], B[k]
        ab = b - a
        L2 = float(ab @ ab)
        if L2 < 1e-9:
            d = np.hypot(P[:, 0] - a[0], P[:, 1] - a[1])
        else:
            t = np.clip(((P - a) @ ab) / L2, 0.0, 1.0)
            proj = a + t[:, None] * ab
            d = np.hypot(P[:, 0] - proj[:, 0], P[:, 1] - proj[:, 1])
        upd = d < best
        best[upd] = d[upd]
        lab[upd] = k
    return best, lab


def _hits(p0, p1, A, B):
    """Which road segments cross the single row segment p0-p1, where, at what angle.

    Distance is the wrong measure for this paper: a road 130 m from the nearest
    shaft is not pressing on the tunnel, and a road 5 m away that runs parallel to
    the row presses on nothing at all. What the mechanics needs is the places
    where the road *crosses* the line the tunnel follows, and the angle there.
    """
    d1 = p1 - p0
    d2 = B - A
    den = d1[0] * d2[:, 1] - d1[1] * d2[:, 0]
    ok = np.abs(den) > 1e-12
    den_safe = np.where(ok, den, 1.0)
    t = ((A[:, 0] - p0[0]) * d2[:, 1] - (A[:, 1] - p0[1]) * d2[:, 0]) / den_safe
    u = ((A[:, 0] - p0[0]) * d1[1] - (A[:, 1] - p0[1]) * d1[0]) / den_safe
    hit = ok & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
    if not hit.any():
        return [], [], []
    H = np.nonzero(hit)[0]
    nr = float(np.hypot(d1[0], d1[1]))
    nb = np.hypot(d2[H][:, 0], d2[H][:, 1])
    cos = np.abs(d2[H] @ d1) / np.maximum(nr * nb, 1e-9)
    ang = np.degrees(np.arccos(np.clip(cos, 0.0, 1.0))).tolist()
    pts = (p0[None, :] + t[H][:, None] * d1[None, :]).tolist()
    return H.tolist(), ang, pts


def _crossings(f, A, B, meta, step=10.0, group=20.0):
    """Road crossings of the row's *fitted curve*, grouped into events.

    Two reasons not to intersect the raw polyline:

    * the polyline joins consecutive shafts, so on a row that runs within 13 m of a
      parallel road the segments zigzag across it and every wobble counts as a
      crossing — Tabas Q01 went from 1 to 11 that way once the second detection
      pass added points on both sides of the carriageway;
    * a dual carriageway is several ways in OSM, so one place where the road passes
      over the tunnel would otherwise count two or three times. Hits within
      ``group`` metres are one event.
    """
    if not len(A) or f["length"] <= 0:
        return []
    ss = np.arange(-f["length"] / 2, f["length"] / 2 + step, step)
    rr = np.polyval(f["coef"], ss)
    C = (f["c"][None, :] + ss[:, None] * f["u"][None, :]
         + rr[:, None] * f["v"][None, :])
    hits = []
    for j in range(len(C) - 1):
        H, aa, pp = _hits(C[j], C[j + 1], A, B)
        for h, a, p in zip(H, aa, pp):
            hits.append((float(p[0]), float(p[1]), float(a), meta[h][0]))
    if not hits:
        return []
    hits.sort(key=lambda t: (t[0], t[1]))
    out = []
    for h in hits:
        if any(math.hypot(h[0] - o["x"], h[1] - o["y"]) <= group for o in out):
            continue
        out.append(dict(x=h[0], y=h[1], ang=h[2], cls=h[3]))
    return out


def _assign_ids(lab, kept, path="data/ir/row_ids.json"):
    """Give each row a stable id, so a row already marked on paper keeps its name.

    Numbering by position in the kept list is only stable while that list does not
    change. It changed: loosening the seeding rule added one row to Tabas, the new
    row took Q04, and the old Q04 became Q05 — a row the co-author had already judged
    under one name silently answered to another. Ids are therefore remembered by
    position on the ground: a row takes the nearest remembered id within 80 m if
    there is one, otherwise the next free number.
    """
    p = Path(path)
    try:
        book = json.load(open(p, encoding="utf-8"))
    except Exception:
        book = {}
    lab_book = book.get(lab, {})
    used = set(lab_book.keys())
    nums = [int(v.split("Q")[-1]) for v in used if "Q" in v]
    nxt = max(nums + [0]) + 1
    taken, out = set(), []
    for r in sorted(kept, key=lambda x: -x["n"]):
        best, bd = None, 80.0
        for cid, (la, lo) in lab_book.items():
            if cid in taken:
                continue
            d = float(np.hypot((r["lat"] - la) * 111320.0,
                               (r["lon"] - lo) * 111320.0 *
                               np.cos(np.radians(la))))
            if d < bd:
                best, bd = cid, d
        if best is None:
            while f"{lab}-Q{nxt:02d}" in used:
                nxt += 1
            best = f"{lab}-Q{nxt:02d}"
            used.add(best)
            nxt += 1
        taken.add(best)
        lab_book[best] = [round(r["lat"], 6), round(r["lon"], 6)]
        r["cid"] = best
        out.append(r)
    book[lab] = lab_book
    p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(book, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return out


def analyse(lab, src, name=None, refine=True):
    """Trace one region's detections and keep the rows that look like chains.

    ``refine`` merges the corridor second pass (qanat/iran_refine.py) into each
    row before it is measured. The cid is assigned from the *first* pass ordering,
    so re-running a region with more shafts never renumbers a row the co-author has
    already marked.
    """
    d = json.load(open(src, encoding="utf-8"))
    pts = d["pts"]
    if not pts:
        return [], {"label": lab, "n_pts": 0, "n_rows": 0, "n_kept": 0}
    P, zone = points_utm(pts)
    stats = {}
    found = rw.extract(P, stats=stats)
    A, B, meta = segments_for(lab, zone=zone)
    Am, Bm, metam = segments_for(lab, zone=zone, classes=MAJOR)
    extra, irf = {}, None
    if refine:
        from . import iran_refine as irf
        extra = irf.load(lab)

    kept = []
    for i, f in enumerate(found, 1):
        cid = f"{lab}-Q{i:02d}"
        Q = P[np.asarray(f["path"], int)]
        if irf is not None and cid in extra and f["n"] >= REFINE_MIN_N:
            Q, f = irf.merge(Q, extra[cid], zone)
        # to_lonlat returns (lon, lat) in that order. Unpacking it as (la, lo) put
        # the longitude in the latitude column of every product, and nothing
        # complained because both are floats in the same range for a while.
        lo, la = (float(v) for v in geo.to_lonlat(Q[:, 0].mean(), Q[:, 1].mean(),
                                                  zone=zone))
        d_road = d_major = float("nan")
        cls = ref = mcls = mref = ""
        if len(A):
            dist, near = pt_seg(Q, A, B)
            d_road = float(dist.min())
            cls, ref = meta[int(near[int(dist.argmin())])]
        if len(Am):
            # distance to a trunk/primary/secondary only: a farm track at 50 m and a
            # highway at 50 m are not the same case, and the nearest-way query
            # almost always returns a tertiary because they are the most numerous
            distm, nearm = pt_seg(Q, Am, Bm)
            d_major = float(distm.min())
            mcls, mref = metam[int(nearm[int(distm.argmin())])]
        (e0, e1) = f["ends"]
        elon, elat = geo.to_lonlat(np.array([e0[0], e1[0]]),
                                   np.array([e0[1], e1[1]]), zone=zone)
        plon, plat = geo.to_lonlat(Q[:, 0], Q[:, 1], zone=zone)
        # does a paved road actually cross this row's line (the tunnel), and at
        # what angle? — the quantity the mechanics of the paper depends on
        ev = _crossings(f, A, B, meta) if len(A) else []
        n_cross = len(ev)
        angs = [e["ang"] for e in ev]
        cross_cls = ev[0]["cls"] if ev else ""
        cross_ll = []
        if ev:
            cxp = np.array([[e["x"], e["y"]] for e in ev], float)
            xlon, xlat = geo.to_lonlat(cxp[:, 0], cxp[:, 1], zone=zone)
            cross_ll = [[round(float(a), 6), round(float(b), 6)]
                        for a, b in zip(np.atleast_1d(xlon), np.atleast_1d(xlat))]
        kept.append(dict(
            label=lab, lat=la, lon=lo,
            n=f["n"], length=f["length"], resid=f["resid"], spacing=f["spacing"],
            sp_cv=f["sp_sd"] / f["spacing"] if f["spacing"] else float("nan"),
            gap_max=f["gap_max"], sag=f["sag"], turn=f["turn"], bearing=f["bearing"],
            lat0=float(elat[0]), lon0=float(elon[0]),
            lat1=float(elat[1]), lon1=float(elon[1]),
            path=[[round(float(a), 6), round(float(b), 6)] for a, b in zip(plon, plat)],
            d_road=d_road, road_cls=cls, road_ref=ref,
            d_major=d_major, major_cls=mcls, major_ref=mref,
            n_cross=n_cross, cross_ang=float(np.median(angs)) if angs else float("nan"),
            cross_cls=cross_cls, cross_ll=cross_ll))
    kept = _assign_ids(lab, kept)
    rej = load_row_verdicts()
    if rej:
        kept = [r for r in kept if r["cid"] not in rej]
    info = {"label": lab, "n_pts": len(pts), "n_segments": stats.get("n_segments", 0),
            "n_rows": stats.get("n_rows", 0), "n_kept": len(kept),
            "n_claimed": stats.get("n_claimed", 0),
            "n_rejected": sum(1 for c in rej if c.startswith(lab + "-")),
            "n_dropped": stats.get("n_segments", 0) - len(kept),
            "utm_zone": int(zone)}
    return kept, info


def run(labels, outdir="data/ir/chains11", figdir="data/ir/regions11"):
    od = Path(outdir); od.mkdir(parents=True, exist_ok=True)
    allr, infos = [], []
    for lab in labels:
        src = Path(figdir) / f"screen_{lab}.json"
        if not src.exists():
            print(f"{lab}: no screening file, skipped")
            continue
        # screening windows share the {lab}-Qnn id space with the block products, so
        # the corridor second pass must not be applied here: its cache is keyed by
        # cid and would be merged into the wrong rows.
        kept, info = analyse(lab, src, refine=False)
        allr += kept
        infos.append(info)
        print(f"{lab}: {info['n_pts']:4d} pts -> {info['n_segments']:3d} segments, "
              f"{info['n_kept']:2d} kept")
    cols = ["cid", "label", "lat", "lon", "n", "length", "resid", "spacing",
            "sp_cv", "gap_max", "sag", "turn", "bearing",
            "d_road", "road_cls", "road_ref", "d_major", "major_cls", "major_ref"]
    with open(od / "chains_ir.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in allr:
            w.writerow({k: _cell(k, v) for k, v in r.items()})
    json.dump(infos, open(od / "screen_summary.json", "w"), ensure_ascii=False, indent=1)
    return allr, infos


if __name__ == "__main__":
    lab = ["R3", "R4", "R7", "R8", "R11"]
    rows, infos = run(lab)
    print(f"\n{len(rows)} chain candidates")
    for r in rows:
        print(f"  {r['cid']}  n={r['n']:3d}  L={r['length']:6.0f} m  "
              f"resid={r['resid']:5.1f}  sp={r['spacing']:5.1f}  "
              f"brg={r['bearing']:5.1f}  road {r['d_road']:6.0f} m")
