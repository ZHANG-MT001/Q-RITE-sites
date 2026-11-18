"""Measured widths of the roads that already exist in the two Turpan study areas.

Why this exists
---------------
The sweep carries a road of $B$ = 26 m, taken from the JTG B01-2014 cross-section
for a four-lane divided expressway. That 26 m is a **design** width for a road
that has not been built; nothing in the two study areas measures 26 m. Between
"a design standard" and "what is on the ground" there is a factor that has
already bitten this project once: at Ardestan the class table said 20-27 m for a
trunk and the pavement turned out to be 8.0 m, because the row crossed one
carriageway of a divided road. The error was only caught because the width was
measured.

This module measures the same way on the Turpan side. Two populations:

``crossing``  the nine places where an existing OSM road crosses a traced chain,
              i.e. the only places in these study areas where a tunnel genuinely
              lies under a road. These are village and county roads.
``class``     points along the widest road class present (G312, tagged trunk and
              given 24.5 m by the class table), sampled on a fixed grid. These do
              not cross a chain; they are here to test the class number itself.

The method is the Iranian one, unchanged: a brightness median over a strip along
the road, a half-depth edge on each side, and the 75 %/25 % cuts as a bracket.
Reusing it is the point - a second operator or a second convention would make the
two countries incomparable.

What the answer turned out to be
--------------------------------
Village roads measure 5-7 m against a class value of 6.5, so the table is right
where the road class is right. G312 measures 5.5-7.0 m against a class value of
24.5: here a `trunk` is a two-lane road, and the table overstates it by a factor
of three and a half. That is the same failure mode as Ardestan, in the opposite
direction, and it is the reason 26 m is described in the manuscript as a design
width rather than as a measurement.

Cases that fail the acceptance rule are kept in the table with the reason. A
width that could not be measured is evidence about the imagery, not about the
road, and dropping it silently would make the sample look cleaner than it is.
"""

from __future__ import annotations

import csv
import json
import textwrap
from pathlib import Path

import numpy as np

from qanat import config, detect, geo, iran_crossing as ix

Z = 19
CACHE = "data/tiles_cache"

# One transect: +-26 m across the road, 32 m of median along it, 0.25 m steps.
SPAN, ALONG, STEP = 26.0, 16.0, 0.25
ALONG_ALT = 12.0            # the stability re-run, as in the Iranian check

# Acceptance, fixed before any case was looked at. A profile has to show a trough
# deep enough and quiet enough that its edge is a measurement and not a guess.
DEPTH_MIN = 20.0            # grey levels between gravel and pavement
BGSTD_MAX = 12.0            # background roughness, |n| in 20-25 m
W_MIN, W_MAX = 3.0, 30.0    # a pavement, not a shadow and not two roads
CTR_MAX = 4.0               # the trough has to straddle the mapped centreline
STAB_MAX = 1.5              # |dW| when the along-road window goes 16 -> 12 m

ROAD_FILE = {"studyA": "data/roads/aoi_b.geojson", "study2": "data/roads/aoi_a.geojson"}

# G312 is sampled on this grid, in metres from the start of the way.
GRID_STEP, GRID_PAD = 200.0, 200.0

COLS = [
    "case", "kind", "site", "road", "highway", "class_width_m", "alpha_deg",
    "measured_m", "w75_m", "w25_m", "centre_m", "bg", "min_grey", "depth",
    "bg_std", "stable_dw_m", "accepted", "reason",
]

_CACHE = {}


# ------------------------------------------------------------------ imagery
def _have(x0, y0, nn, z=Z):
    """True only if every tile of the block is already on disk.

    A missing tile makes ``fetch_tile`` reach for the network and retry for
    seconds; on a 100-point grid that is the difference between two minutes and
    an hour. Skipping is also the honest behaviour: a case with no imagery was
    not measured, and must not be scored as one that was.
    """
    root = Path(CACHE) / str(z)
    return all((root / str(x0 + i) / f"{y0 + j}.jpg").exists()
               for i in range(nn) for j in range(nn))


def _profile(c, t, along=ALONG):
    key = (round(float(c[0]), 1), round(float(c[1]), 1),
           round(float(t[0]), 3), round(float(t[1]), 3), along)
    if key in _CACHE:
        return _CACHE[key]

    nrm = np.array([-t[1], t[0]])
    ns = np.arange(-SPAN, SPAN + 1e-9, STEP)
    us = np.arange(-along, along + 1e-9, 2.0)
    P = np.array([c + u * t + s * nrm for u in us for s in ns])
    lon, lat = geo.to_lonlat(P[:, 0], P[:, 1])
    tx, ty = ix._vtile(lon, lat, Z)
    # Only the tiles the sample actually touches, plus one on the far side for the
    # bilinear step. Padding the box by three tiles, as the single-case code does,
    # turns a 2x2 requirement into an 8x8 one and throws away half the grid.
    x0, y0 = int(tx.min()), int(ty.min())
    nn = int(max(tx.max() - x0, ty.max() - y0)) + 2
    if not _have(x0, y0, nn):
        _CACHE[key] = None
        return None

    img, _ = detect.block(x0, y0, nn, Z, cache=CACHE)
    g = img[..., :3].mean(axis=2).astype(float)
    x, y = (tx - x0) * 256, (ty - y0) * 256
    xi = np.clip(np.floor(x).astype(int), 1, g.shape[1] - 3)
    yi = np.clip(np.floor(y).astype(int), 1, g.shape[0] - 3)
    fx, fy = x - xi, y - yi
    v = (g[yi, xi] * (1 - fx) * (1 - fy) + g[yi, xi + 1] * fx * (1 - fy)
         + g[yi + 1, xi] * (1 - fx) * fy + g[yi + 1, xi + 1] * fx * fy)
    prof = np.median(v.reshape(len(us), len(ns)), axis=0)
    out = (ns, np.convolve(prof, np.ones(11) / 11, mode="same"))
    _CACHE[key] = out
    return out


def _edges(ns, p, frac, near=20.0, band=(20.0, 25.0), reach=24.0):
    """Left and right edge of the trough, at a fraction of its depth."""
    core = np.abs(ns) <= near
    j = int(np.nonzero(core)[0][np.argmin(p[core])])
    bgb = (np.abs(ns) >= band[0]) & (np.abs(ns) <= band[1])
    bg = float(np.percentile(p[bgb], 85))
    thr = bg - frac * (bg - p[j])
    lim = np.nonzero(np.abs(ns) <= reach)[0]
    i0, i1 = int(lim[0]), int(lim[-1])
    a = b = j
    while a > i0 and p[a - 1] < thr:
        a -= 1
    while b < i1 and p[b + 1] < thr:
        b += 1
    return float(ns[a]), float(ns[b]), bg, float(p[j]), float(np.std(p[bgb]))


# ------------------------------------------------------------------- cases
def _at(W, s):
    """Point and tangent at arclength s along a polyline."""
    L = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(W, axis=0).T))])
    k = int(np.searchsorted(L, s) - 1)
    k = max(0, min(k, len(W) - 2))
    f = (s - L[k]) / max(L[k + 1] - L[k], 1e-9)
    t = W[k + 1] - W[k]
    return W[k] + f * (W[k + 1] - W[k]), t / max(float(np.hypot(*t)), 1e-9)


def _aoi(site, pad=1000.0):
    """Bounding box of the traced chains, expanded by ``pad`` metres.

    Class points outside it are not part of this study and, more practically,
    have no cached imagery: sampling the whole of G312 would add 350 rows that
    say only 'not here'.
    """
    ch = json.load(open(f"data/chains/{site}_chains.geojson", encoding="utf-8"))
    P = []
    for f in ch["features"]:
        if f["properties"].get("kind") != "chain":
            continue
        P += f["geometry"]["coordinates"]
    if not P:
        raise ValueError(f"no chains for {site}")
    e, n = geo.to_utm(np.array(P)[:, 0], np.array(P)[:, 1])
    return (e.min() - pad, e.max() + pad, n.min() - pad, n.max() + pad)


def _roads():
    out = {}
    for site, p in ROAD_FILE.items():
        feats = json.load(open(p, encoding="utf-8"))["features"]
        for f in feats:
            c = np.asarray(f["geometry"]["coordinates"], float)
            if c.ndim != 2 or len(c) < 2:
                continue
            e, n = geo.to_utm(c[:, 0], c[:, 1])
            out[f["properties"].get("osm_id")] = (site, f["properties"],
                                                  np.column_stack([e, n]))
    return out


def cases():
    """The two populations, in a fixed order so the table is reproducible."""
    R = _roads()
    out = []

    with open("data/out/events_roads.csv", encoding="utf-8-sig") as f:
        crossed = list(csv.DictReader(f))
    for r in crossed:
        site, prop, W = R[r["road"]]
        i = int(r["seg"])
        a, b = W[i], W[min(i + 1, len(W) - 1)]
        t = b - a
        out.append(dict(
            case=f"X-{r['cid']}", kind="crossing", site=site,
            road=r["road"], highway=r["highway"],
            class_width_m=float(r["width_m"]),
            alpha_deg=float(r["alpha_deg"]),
            c=np.array([float(r["x_m"]), float(r["y_m"])]),
            t=t / max(float(np.hypot(*t)), 1e-9)))
    out.sort(key=lambda d: (d["site"], d["case"]))

    # The widest class present in either area, sampled on a fixed distance grid.
    seen = set()
    for osm, (site, prop, W) in sorted(R.items()):
        hw = str(prop.get("highway", "")).lower()
        if hw not in ("trunk", "primary") or osm in seen:
            continue
        seen.add(osm)
        e0, e1, n0, n1 = _aoi(site)
        L = float(np.sum(np.hypot(*np.diff(W, axis=0).T)))
        for s in np.arange(GRID_PAD, L - GRID_PAD, GRID_STEP):
            c, t = _at(W, s)
            if not (e0 <= c[0] <= e1 and n0 <= c[1] <= n1):
                continue
            out.append(dict(
                case=f"C-{osm.split('/')[-1]}-{int(round(s))}", kind="class",
                site=site, road=osm, highway=hw,
                class_width_m=float(prop.get("width_m")
                                    or config.ROAD_WIDTH[hw][1]),
                alpha_deg=float("nan"), c=c, t=t))
    return out


def measure(case, *, profile_fn=_profile):
    """Measure a case from cached imagery or a supplied profile function."""
    prof = profile_fn(case["c"], case["t"])
    row = dict(case=case["case"], kind=case["kind"], site=case["site"],
               road=case["road"], highway=case["highway"],
               class_width_m=round(case["class_width_m"], 1),
               alpha_deg=(round(case["alpha_deg"], 2)
                          if case["alpha_deg"] == case["alpha_deg"] else ""))
    if prof is None:
        row.update(measured_m="", w75_m="", w25_m="", centre_m="", bg="",
                   min_grey="", depth="", bg_std="", stable_dw_m="",
                   accepted=0, reason="no imagery cached at this point")
        return row

    ns, p = prof
    lo, hi, bg, mn, sd = _edges(ns, p, 0.5)
    a75, b75, _, _, _ = _edges(ns, p, 0.75)
    a25, b25, _, _, _ = _edges(ns, p, 0.25)
    w = hi - lo
    alt = profile_fn(case["c"], case["t"], along=ALONG_ALT)
    dw = (abs(_edges(*alt, 0.5)[1] - _edges(*alt, 0.5)[0] - w) if alt else float("nan"))

    why = []
    if bg - mn < DEPTH_MIN:
        why.append(f"depth {bg - mn:.0f} < {DEPTH_MIN:.0f}")
    if sd > BGSTD_MAX:
        why.append(f"background std {sd:.0f} > {BGSTD_MAX:.0f}")
    if not (W_MIN <= w <= W_MAX):
        why.append(f"width {w:.1f} outside {W_MIN:.0f}-{W_MAX:.0f} m")
    if abs((lo + hi) / 2.0) > CTR_MAX:
        why.append(f"trough centre {(lo + hi) / 2.0:+.1f} m off the mapped line")
    if dw != dw or dw > STAB_MAX:
        why.append(f"width moves {dw:.1f} m with the along-road window")

    row.update(measured_m=round(w, 1), w75_m=round(b75 - a75, 1),
               w25_m=round(b25 - a25, 1), centre_m=round((lo + hi) / 2.0, 1),
               bg=round(bg), min_grey=round(mn), depth=round(bg - mn),
               bg_std=round(sd, 1), stable_dw_m=round(dw, 1),
               accepted=0 if why else 1, reason="; ".join(why) if why else "")
    return row


def run(out="data/out/width_transect.csv"):
    rows = [measure(c) for c in cases()]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    return rows


def read(path="data/out/width_transect.csv"):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


# ------------------------------------------------------------------ checks
def check(rows):
    """Fail on any row whose verdict does not follow from its own numbers."""
    bad = []
    for r in rows:
        if r["measured_m"] == "":
            if int(r["accepted"]) or not r["reason"]:
                bad.append(f"{r['case']}: unmeasured but accepted / unexplained")
            continue
        w, d, sd = float(r["measured_m"]), float(r["depth"]), float(r["bg_std"])
        ctr, dw = float(r["centre_m"]), float(r["stable_dw_m"])
        ok = (d >= DEPTH_MIN and sd <= BGSTD_MAX and W_MIN <= w <= W_MAX
              and abs(ctr) <= CTR_MAX and dw <= STAB_MAX)
        if ok != bool(int(r["accepted"])):
            bad.append(f"{r['case']}: accepted={r['accepted']} contradicts its numbers")
        if not ok and not r["reason"]:
            bad.append(f"{r['case']}: rejected without a reason")
    if bad:
        raise AssertionError("width transect verdicts do not follow from the numbers:\n  - "
                             + "\n  - ".join(bad))

    acc = [r for r in rows if int(r["accepted"])]
    x = [r for r in acc if r["kind"] == "crossing"]
    k = [r for r in acc if r["kind"] == "class"]
    probe = []
    if len(x) < 4:
        probe.append(f"only {len(x)} village crossings measured; the sample is too thin")
    if not k:
        probe.append("no class-case measured, so the class width is untested")
    vm = np.median([float(r["measured_m"]) for r in x]) if x else float("nan")
    if x and not (4.0 <= vm <= 9.0):
        probe.append(f"village median {vm:.1f} m is not a village road")
    # The sentence the section exists for: on this network the class width is not
    # a measurement. A median rule, not a per-point one, because a single junction
    # widens the road and is not a counter-example.
    if k:
        ratios = [float(r["measured_m"]) / float(r["class_width_m"]) for r in k]
        med = float(np.median(ratios))
        if med > 0.5:
            probe.append(f"median measured/class width {med:.2f} > 0.5, so "
                         f"'the table overstates a trunk here' would be false")
        if max(ratios) > 0.75:
            probe.append(f"one point reaches {max(ratios):.2f} of its class width, so "
                         f"'no measured road reaches its class value' would be false")
    if probe:
        raise AssertionError("width transect checks failed:\n  - " + "\n  - ".join(probe))
    return True


# --------------------------------------------------------------- rendering
def _fill(t):
    return textwrap.fill(" ".join(t.split()), width=88)


def render(rows):
    acc = [r for r in rows if int(r["accepted"])]
    rej = [r for r in rows if not int(r["accepted"])]
    x = sorted([r for r in acc if r["kind"] == "crossing"], key=lambda r: r["case"])
    k = [r for r in acc if r["kind"] == "class"]
    kmeas = [r for r in rows if r["kind"] == "class" and r["measured_m"] != ""]
    n_noimg = sum(1 for r in rows if r["measured_m"] == "")

    head = ("| case | area | OSM class | class width (m) | measured (m) "
            "| 75-25 % bracket (m) | centre offset (m) | trough depth | "
            "stability (m) |")
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for r in x:
        lines.append(
            f"| {r['case']} | {r['site']} | {r['highway']} | "
            f"{float(r['class_width_m']):.1f} | {float(r['measured_m']):.1f} | "
            f"{float(r['w75_m']):.1f}-{float(r['w25_m']):.1f} | "
            f"{float(r['centre_m']):+.1f} | {int(float(r['depth']))} | "
            f"{float(r['stable_dw_m']):.1f} |")
    cross_tbl = "\n".join(lines)

    vm = np.median([float(r["measured_m"]) for r in x]) if x else float("nan")
    vc = np.median([float(r["class_width_m"]) for r in x]) if x else float("nan")

    if k:
        kw = np.array([float(r["measured_m"]) for r in k])
        kc = float(np.median([float(r["class_width_m"]) for r in k]))
        q1, q3 = np.percentile(kw, [25, 75])
        kmed, kmax = float(np.median(kw)), float(kw.max())
    else:
        kw = np.array([])
        kc, q1, q3, kmed, kmax = float("nan"), float("nan"), float("nan"), float("nan"), float("nan")

    cls_head = ("| OSM class | class width (m) | points with imagery | points accepted "
                "| median (m) | interquartile (m) | widest (m) | widest / class |")
    cls_tbl = "\n".join([
        cls_head, "|" + "---|" * (cls_head.count("|") - 1),
        f"| {k[0]['highway'] if k else 'trunk'} | {kc:.1f} | {len(kmeas)} | {len(k)} "
        f"| {kmed:.1f} | {q1:.1f}-{q3:.1f} | {kmax:.1f} | {kmax / kc:.2f} |",
    ])

    def tally(kind_word):
        n = sum(1 for r in rej if kind_word in r["reason"])
        return n

    paras = [
        "The transect used in Iran is applied to the roads that already exist in the two "
        "Turpan study areas. Nothing about the method changes: a brightness median over a "
        "32 m strip along the road, a half-edge on each side at half the profile depth, and "
        "the 75 % and 25 % cuts as a bracket. Reusing the convention is the point, because "
        "a second operator would make the two countries incomparable.",

        "### Roads that actually cross a traced chain",

        _fill(f"Nine OSM ways cross a traced chain in these two areas, and those are the only "
              f"places where a tunnel lies under a road at all. {len(x)} of the nine yield a "
              f"transect that passes the acceptance rule. Their measured pavement has a "
              f"median of {vm:.1f} m against a median class value of {vc:.1f} m, so on these "
              f"small roads the table is right, and the method transfers to a second country "
              f"and a second imagery mosaic without a change of convention."),

        cross_tbl,

        _fill("The 75 %/25 % bracket is wide wherever the pavement edge grades into a dust "
              "shoulder rather than ending at a kerb, which is most of these cases. The "
              "reported width is the half-depth cut; the bracket is carried so that a reader "
              "can see how much the answer depends on where the edge is called, and it is the "
              "same convention the Iranian numbers use, so the two are comparable even where "
              "both are uncertain."),

        "### The widest class present, sampled away from any chain",

        _fill(f"G312 is the only trunk in the study areas, and the class table gives it 24.5 m "
              f"from JTG B01-2014 on the assumption that a trunk is a four-lane divided road. "
              f"Within the study bounding box it was sampled every 200 m along its length: "
              f"{len(kmeas)} points fall inside cached imagery, {len(k)} pass the acceptance "
              f"rule, and the measured pavement has a median of {kmed:.1f} m, an interquartile "
              f"range of {q1:.1f}-{q3:.1f} m, and a widest value of {kmax:.1f} m, which is "
              f"{kmax / kc:.0%} of the class width. No measured point reaches its class value."),

        cls_tbl,

        _fill("That is the same failure as Ardestan, in the opposite direction. There the class "
              "table named a divided road and the row crossed one carriageway of it; here it "
              "names a four-lane formation and the road on the ground is two lanes. In both "
              "cases the class number was wrong by a factor of three or more and nothing "
              "downstream could have caught it, because a wider road only makes the exposure "
              "figure look more like the one the model expects. The difference is that here "
              "the width was measured, so the error stays in the table and does not reach the "
              "result."),

        "### Cases that were not measured",

        _fill(f"{len(rej)} points are excluded and kept in the csv with their reason. "
              f"{n_noimg} of them ({n_noimg / max(len(rows), 1):.0%}) simply fall outside the "
              f"cached imagery, and the rest were measured and then rejected. The reasons "
              f"overlap, because a point can fail more than one rule: {tally('background std')} "
              f"fail on a background too rough for a half-depth edge to mean anything, "
              f"{tally('trough centre')} because the trough does not straddle the mapped "
              f"centreline, which usually means the OSM line is offset by more than the road is "
              f"wide, and {tally('width moves')} because the width moves when the along-road "
              f"window changes. Reporting them matters: a width that could not be measured is "
              f"evidence about the imagery or the site, not about the road, and a sample that "
              f"lists only its successes cannot be audited."),

        _fill("What this does not establish is a formation width for the planned expressway. No "
              "such road exists in either area, so the 26 m in the sweep stays a design value "
              "and is described as one. What the measurements do establish is narrower and "
              "sufficient: the class table is reliable where the road is small and unreliable "
              "where it is large, so a paper carrying a road of a given width has to say which "
              "kind of number that width is."),
    ]

    out = []
    for p in paras:
        if p.startswith(("#", "|")):
            out.append(p)
        else:
            out.append(_fill(p))
    return "\n\n".join(out) + "\n"

def write(rows, outdir="paper", csv_path="data/out/width_transect.csv"):
    (Path(outdir) / "width_transect.md").write_text(render(rows), encoding="utf-8")
    return Path(csv_path), Path(outdir) / "width_transect.md"


def verify(outdir="paper", csv_path="data/out/width_transect.csv"):
    rows = read(csv_path)
    want = render(rows)
    got = (Path(outdir) / "width_transect.md").read_text(encoding="utf-8")
    if want != got:
        raise AssertionError(f"{outdir}/width_transect.md does not match a fresh render "
                             f"of {csv_path}")
    return True


def main(outdir="paper", csv_path="data/out/width_transect.csv"):
    rows = run(csv_path)
    check(rows)
    write(rows, outdir=outdir, csv_path=csv_path)
    verify(outdir, csv_path)
    acc = [r for r in rows if int(r["accepted"])]
    print(f"width transect: {len(acc)}/{len(rows)} points measured")
    print(f"  {'case':18s} {'kind':9s} {'class':>6s} {'meas':>6s}  reason")
    for r in rows:
        if int(r["accepted"]):
            print(f"  {r['case']:18s} {r['kind']:9s} {r['class_width_m']:>6} "
                  f"{r['measured_m']:>6}  ok")
    for r in rows:
        if not int(r["accepted"]):
            print(f"  {r['case']:18s} {r['kind']:9s} {r['class_width_m']:>6} "
                  f"{'--':>6}  {r['reason'][:56]}")
    print(f"written {csv_path}, {outdir}/width_transect.md")
    return rows
