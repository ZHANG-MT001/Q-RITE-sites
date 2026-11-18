"""Single-case geometry of the two places where a paved road crosses a traced row.

Turpan produced the distribution: 7467 crossings, loaded length and exposed shaft
mouths as a function of crossing angle. Iran can only contribute existence and two
worked examples, and this module computes exactly those two, from evidence that
holds up at the resolution available.

Widths are measured, and the measurement is a transect. Each crossing is sampled
perpendicular to the carriageway, with the brightness median taken over 32 m along
the road so that pits, shadows and sand streaks do not enter; the pavement is the
trough that remains, 45-60 per cent of the gravel's brightness with a depth of
60-73 grey levels. The edge is taken at half depth - where the reflected brightness
has returned halfway to the gravel - and that convention is what makes the number
reproducible. It is stable: widening the along-road window from 24 m to 32 m moves
the width by 0.3 m at both crossings.

Two earlier attempts failed because they sampled along the row instead of across the
road. At 74 degrees the two directions differ by only 16 degrees, the transect then
runs down the carriageway instead of across it, and a flat profile looks like a
surface without an edge. Running parallel to the thing you are trying to bound is
the one error that produces a confident negative.

The measured widths also correct a classification mistake. Both rows were matched to
a road class and given that class's width - 8-10 m for the Tabas primary, 20-27 m
for the Ardestan trunk, on the assumption that a trunk here is a dual carriageway.
The transect says the Ardestan pavement is 8 m, not 20-27: the row crosses one
carriageway of the dual, and the other runs about 100 m away, outside the shaft gap.
Using the class width would have overstated the loaded length at that site by a
factor of nearly three, and the error would not have shown up anywhere, because a
wider road only makes the exposure figure look more like Turpan's.

The gap is measured on each side of the pavement separately. That distinction came
out of a test meant to be a formality: buffering the centreline by half the width
contains no detected shaft at either site, which is expected - the shafts under a
pavement are the ones that are buried and therefore undetectable - so the buffer
cannot validate the mouth count, only restate that it is an expectation. What the
same test showed is that the Tabas gap is lopsided: 54 m on one side of the crossing
and 13 m on the other, which an 11 m road cannot explain. At 0.25 m/px a second,
parallel track crosses the row inside the long side (the co-author recorded it in the
field as a village road). The gap there is the union of two obstacles and says
nothing about the primary alone, so only its short side is used. Ardestan's gap is
26 and 19 m, symmetric enough to be used whole.

The single number this section exists for is the loaded length, L = W / sin a. At
71-74 deg the crossing is 16-19 deg off perpendicular, so sin a is 0.95-0.96 and the
tunnel under the road is almost exactly one carriageway wide. That puts both Iranian
cases at the perpendicular end-member, where Turpan's exposure curve has its minimum.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from qanat import detect, geo, iran_chains as ic

# Width by class, kept only as a cross-check on the measurement. Primary: two 3.5 m
# lanes plus shoulders. Trunk: two 7-7.5 m carriageways with a median, which is what
# the class means but not what a row crossing one carriageway sees.
CLASS_WIDTH = {"primary": (8.0, 10.0), "trunk": (20.0, 27.0),
               "secondary": (7.0, 9.0), "motorway": (24.0, 32.0)}

# A gap this lopsided is not centred on the road it is supposed to describe.
ASYM_MIN = 0.5

# Obstacles other than the class road that are known to sit inside a gap. R4-Q01's
# entry rests on two independent observations - the parallel track is visible at
# 0.25 m/px inside the long side, and the co-author recorded a village road crossing
# this row - and the asymmetry ratio flags the same case on its own.
SECOND_FEATURE = {
    "R4-Q01": "parallel track crossing inside the south-west side",
}

CASES = [("R4", "R4-Q01"), ("R6", "R6-Q01")]

TRANSECT = dict(z=19, span=26.0, along=16.0, step=0.25)

_CACHE = {}


def _vtile(lon, lat, z):
    lon = np.asarray(lon, float)
    lat = np.radians(np.asarray(lat, float))
    return ((lon + 180.0) / 360.0 * (2 ** z),
            (1.0 - np.log(np.tan(lat) + 1 / np.cos(lat)) / np.pi) / 2.0 * (2 ** z))


def _row_frame(row, zone):
    """Row points in UTM, the crossing point, and the row's heading there."""
    P = np.array(row["path"], float)
    E, N = geo.to_utm(P[:, 0], P[:, 1], zone=zone)
    Q = np.column_stack([E, N])
    lon, lat = row["cross_ll"][0]
    X, Y = geo.to_utm(np.array([lon]), np.array([lat]), zone=zone)
    c = np.array([float(np.atleast_1d(X)[0]), float(np.atleast_1d(Y)[0])])
    k = int(np.argmin(np.hypot(*(Q - c).T)))
    d = Q[min(k + 1, len(Q) - 1)] - Q[max(k - 1, 0)]
    return Q, c, d / max(float(np.hypot(*d)), 1e-9)


def _road_at(lab, c, zone):
    """Nearest paved way to the crossing: distance, its vertices, its segment.

    pt_seg returns one distance per query point and, separately, the index of the
    nearest segment. Reading the returned index as a per-point array once put a
    transect on a segment 38 km away, and the profile came back flat; both returns
    are used here as what they are.
    """
    best = None
    for w in json.load(open(f"data/ir/roads11/{lab}.json")):
        if w["cls"] not in CLASS_WIDTH or len(w["pt"]) < 2:
            continue
        P = np.array([[g[1], g[0]] for g in w["pt"]])
        E, N = geo.to_utm(P[:, 0], P[:, 1], zone=zone)
        W = np.column_stack([E, N])
        dist, near = ic.pt_seg(c[None, :], W[:-1], W[1:])
        if best is None or dist[0] < best[0]:
            best = (float(dist[0]), W, int(near[0]), w["cls"])
    dist, W, k, cls = best
    a, b = W[k], W[k + 1]
    t = (b - a) / max(float(np.hypot(*(b - a))), 1e-9)
    return dict(dist=dist, W=W, k=k, cls=cls, t=t,
                n=np.array([-t[1], t[0]]), seg_len=float(np.hypot(*(b - a))))


def _transect(lab, cid):
    """Brightness across the carriageway, sampled perpendicular to it."""
    key = ("t", lab, cid)
    if key in _CACHE:
        return _CACHE[key]
    m = measure(lab, cid, _depth=1)      # geometry only, no transect of its own
    row = m["_row"]
    zone = m["_zone"]
    c = m["_c"]
    R = m["_road"]
    t, nrm = R["t"], R["n"]
    ns = np.arange(-TRANSECT["span"], TRANSECT["span"] + 1e-9, TRANSECT["step"])
    us = np.arange(-TRANSECT["along"], TRANSECT["along"] + 1e-9, 2.0)
    P = np.array([c + u * t + s * nrm for u in us for s in ns])
    lon, lat = geo.to_lonlat(P[:, 0], P[:, 1], zone=zone)
    tx, ty = _vtile(lon, lat, TRANSECT["z"])
    x0, y0 = int(tx.min()) - 3, int(ty.min()) - 3
    nn = int(max(tx.max() - x0, ty.max() - y0)) + 6
    img, _ = detect.block(x0, y0, nn, TRANSECT["z"], cache="data/tiles_cache/ir")
    g = img[..., :3].mean(axis=2) if img.ndim == 3 else img.astype(float)
    x, y = (tx - x0) * 256, (ty - y0) * 256
    xi = np.clip(np.floor(x).astype(int), 1, g.shape[1] - 3)
    yi = np.clip(np.floor(y).astype(int), 1, g.shape[0] - 3)
    fx, fy = x - xi, y - yi
    v = (g[yi, xi] * (1 - fx) * (1 - fy) + g[yi, xi + 1] * fx * (1 - fy)
         + g[yi + 1, xi] * (1 - fx) * fy + g[yi + 1, xi + 1] * fx * fy)
    prof = np.median(v.reshape(len(us), len(ns)), axis=0)
    prof = np.convolve(prof, np.ones(11) / 11, mode="same")
    _CACHE[key] = (ns, prof)
    return ns, prof


def _edges(ns, p, frac, near=18.0, band=(18.0, 25.0), reach=23.0):
    """Left and right edge of the trough, at a given fraction of its depth."""
    core = np.abs(ns) <= near
    j = int(np.nonzero(core)[0][np.argmin(p[core])])
    bg = float(np.percentile(p[(np.abs(ns) >= band[0]) & (np.abs(ns) <= band[1])], 85))
    thr = bg - frac * (bg - p[j])
    lim = np.nonzero(np.abs(ns) <= reach)[0]
    i0, i1 = int(lim[0]), int(lim[-1])
    a = b = j
    while a > i0 and p[a - 1] < thr:
        a -= 1
    while b < i1 and p[b + 1] < thr:
        b += 1
    return float(ns[a]), float(ns[b]), bg, float(p[j])


def width(lab, cid):
    """Measured carriageway width: half-depth value plus the two bracketing cuts."""
    key = ("w", lab, cid)
    if key in _CACHE:
        return _CACHE[key]
    ns, p = _transect(lab, cid)
    lo, hi, bg, mn = _edges(ns, p, 0.5)
    a75, b75, _, _ = _edges(ns, p, 0.75)
    a25, b25, _, _ = _edges(ns, p, 0.25)
    out = dict(fwhm=hi - lo, w75=b75 - a75, w25=b25 - a25,
               centre=(lo + hi) / 2.0, bg=bg, min=mn, depth=bg - mn)
    _CACHE[key] = out
    return out


def measure(lab, cid, _depth=0):
    """Geometry of one crossing, from the traced row, the road and the transect."""
    kept, _ = ic.analyse(lab, f"data/ir/blocks11/block_{lab}.json", refine=True)
    row = next(r for r in kept if r["cid"] == cid)
    _, zone = ic.points_utm([{"lat": v, "lon": u} for u, v in row["path"]])
    Q, c, d = _row_frame(row, zone)
    R = _road_at(lab, c, zone)
    wid = None if _depth else width(lab, cid)
    if wid is None:                       # called from _transect, which needs only geometry
        m = dict(cid=cid, site=lab, cls=R["cls"], road_ref=row["road_ref"],
                 _row=row, _zone=zone, _c=c, _road=R)
        return m

    # the pavement centre, not the mapping centreline, is where the road actually is
    c_ref = c + R["n"] * wid["centre"]
    s = np.sort((Q - c_ref) @ d)
    i = int(np.searchsorted(s, 0.0))
    lo, hi = float(s[i - 1]), float(s[i])
    span = hi - lo
    sp = max(float(row["spacing"]), 1e-9)

    sin_a = math.sin(math.radians(row["cross_ang"]))
    L, L75, L25 = wid["fwhm"] / sin_a, wid["w75"] / sin_a, wid["w25"] / sin_a

    # Each side of the pavement is a separate measurement: how far the row runs
    # without a shaft before it on one side and after it on the other. Only their
    # sum is the gap; their difference is what identifies a second cause.
    side_low, side_high = abs(lo), hi
    asym = min(side_low, side_high) / max(side_low, side_high, 1e-9)
    # margin beyond the pavement edge, on the half-depth width that the table
    # reports; a bracket here would only restate the width bracket two columns over
    apron_low = side_low - L / 2.0
    apron_high = side_high - L / 2.0

    cls_lo, cls_hi = CLASS_WIDTH.get(R["cls"], (7.0, 10.0))
    clean = (cid not in SECOND_FEATURE) and asym >= ASYM_MIN
    note = SECOND_FEATURE.get(cid, "")
    if not clean and not note:
        note = "gap not centred on the crossing"
    if wid["fwhm"] < 0.6 * cls_lo:
        note = (note + "; " if note else "") + \
            f"row crosses one carriageway of the {R['cls']}, not the whole road"

    return dict(
        cid=cid, site=lab, cls=R["cls"], road_ref=row["road_ref"],
        angle=float(row["cross_ang"]), n=row["n"], spacing=sp,
        length=float(row["length"]), span=span, span_spacings=span / sp,
        side_low=side_low, side_high=side_high, asym=asym,
        width=wid["fwhm"], width_lo=wid["w75"], width_hi=wid["w25"],
        width_centre=wid["centre"], depth=wid["depth"], bg=wid["bg"],
        class_lo=cls_lo, class_hi=cls_hi,
        loaded=L, loaded_lo=L75, loaded_hi=L25,
        excess_lo=span - L25, excess_hi=span - L75,
        apron_low=apron_low, apron_high=apron_high,
        mouths=L / sp, mouths_lo=L75 / sp, mouths_hi=L25 / sp,
        clean=clean, note=note,
        before=i, after=len(s) - i,
        pos_frac=float(-s.min() / max(s.max() - s.min(), 1e-9)),
        span_lo=lo, span_hi=hi, cross_ll=row["cross_ll"][0],
    )


COLS = ["cid", "site", "road_class", "road_ref", "angle_deg",
        "paved_width_m", "paved_width_75_m", "paved_width_25_m",
        "transect_depth_grey", "paved_centre_offset_m",
        "class_width_lo_m", "class_width_hi_m",
        "loaded_length_m", "loaded_length_lo_m", "loaded_length_hi_m",
        "shaft_mouths", "shaft_mouths_lo", "shaft_mouths_hi",
        "shaft_free_span_m", "span_in_spacings",
        "gap_low_side_m", "gap_high_side_m", "asym_ratio",
        "gap_minus_pavement_lo_m", "gap_minus_pavement_hi_m",
        "apron_low_m", "apron_high_m",
        "gap_is_road_only", "shaft_spacing_m", "row_length_m",
        "shafts_before", "shafts_after", "crossing_position_frac", "note"]


def _row_out(r):
    return [r["cid"], r["site"], r["cls"], r["road_ref"], round(r["angle"], 1),
            round(r["width"], 1), round(r["width_lo"], 1), round(r["width_hi"], 1),
            round(r["depth"], 0), round(r["width_centre"], 1),
            r["class_lo"], r["class_hi"],
            round(r["loaded"], 1), round(r["loaded_lo"], 1), round(r["loaded_hi"], 1),
            round(r["mouths"], 2), round(r["mouths_lo"], 2), round(r["mouths_hi"], 2),
            round(r["span"], 1), round(r["span_spacings"], 2),
            round(r["side_low"], 1), round(r["side_high"], 1), round(r["asym"], 2),
            round(r["excess_lo"], 1), round(r["excess_hi"], 1),
            round(r["apron_low"], 1), round(r["apron_high"], 1),
            int(r["clean"]), round(r["spacing"], 1), round(r["length"], 0),
            r["before"], r["after"], round(r["pos_frac"], 2), r["note"]]


def run(cases=CASES, out="data/ir/iran_crossings.csv"):
    rows = [measure(lab, cid) for lab, cid in cases]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        for r in rows:
            w.writerow(_row_out(r))
    for r in rows:
        print(f"{r['cid']}: {r['cls']} {r['angle']:.0f} deg · pavement "
              f"{r['width']:.1f} m [{r['width_lo']:.1f}-{r['width_hi']:.1f}], "
              f"depth {r['depth']:.0f}, centre {r['width_centre']:+.1f} m "
              f"(class says {r['class_lo']:.0f}-{r['class_hi']:.0f})")
        print(f"   L {r['loaded']:.1f} m [{r['loaded_lo']:.1f}-{r['loaded_hi']:.1f}], "
              f"mouths {r['mouths']:.2f} [{r['mouths_lo']:.2f}-{r['mouths_hi']:.2f}]")
        print(f"   gap {r['span']:.1f} m = {r['side_low']:.1f} + {r['side_high']:.1f}, "
              f"asym {r['asym']:.2f}, road-only {'yes' if r['clean'] else 'NO'}")
        if r["note"]:
            print(f"   note: {r['note']}")
    return rows


if __name__ == "__main__":
    run()
