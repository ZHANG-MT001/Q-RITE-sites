"""Crossing-angle dataset: one row per (plan, chain) crossing.

The mechanics layer is not implemented here and will not be. What this module
produces is the geometry that a mechanics model needs and nothing more: the
angle at which an alignment meets a chain, the length of tunnel the road
formation actually sits over, how many shaft mouths that length contains, and
where along the chain the crossing falls.

Three quantities, deliberately kept apart:

``coupling_m``      the length of chain under the road formation itself,
                    B/sin(a), where B is the formation width from
                    ``config.ROAD_WIDTH`` (not the paved width - see the note
                    there and ``qanat width``). This is what a load-transfer
                    model should be fed: it is the run over which wheel loads
                    act directly on the crown.
``in_corridor_m``   the length of the same chain inside the +-hw corridor. This
                    is the exposure measure used by ``scan``/``cases``, so it is
                    the one that ties back to Lq. It is longer than coupling_m
                    by 2hw/B.
``n_open_*``        how many shaft mouths those two lengths contain. A shaft
                    mouth is a hole through the crown, so this is the count of
                    places where the load path has no roof at all.

Nothing here assumes a burial depth. Depth enters downstream, from the value
the mechanics group supplies.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from . import cases, config, geo, profile

HW = 100.0                                  # corridor half width, m (scan default)
B_ROAD = config.ROAD_WIDTH["motorway"][1]   # 26.0 m, expressway formation width

COLS = [
    "site", "plan", "bearing_deg", "offset_m", "cid",
    "alpha_deg", "n_seg",
    "coupling_m", "in_corridor_m", "spacing_m", "n_open_road", "n_open_corr",
    "chain_len_m", "n_shaft", "grade", "active_share", "pos_along", "ew_frac",
    "x_m", "y_m",
]

PLAN_COLS = [
    "site", "plan", "bearing_deg", "offset_m", "rel_deg",
    "n_events", "coupling_sum", "n_open_road_sum", "in_corridor_sum",
    "alpha_min", "alpha_med",
]


def _unit(v):
    n = float(np.hypot(*v))
    return v / n if n > 1e-12 else np.array([1.0, 0.0])


def _fold_deg(u, v):
    """Angle between two undirected axes, in [0, 90]."""
    c = abs(float(u @ v)) / max(float(np.hypot(*u) * np.hypot(*v)), 1e-12)
    return float(math.degrees(math.acos(min(1.0, max(-1.0, c)))))


def load_site(site, base="data"):
    """Chains (UTM axes) and shafts (UTM points + state) for one study area."""
    b = Path(base)
    ch = json.load(open(b / f"chains/{site}_chains.geojson", encoding="utf-8"))
    feats = [f for f in ch["features"] if f["properties"].get("kind") == "chain"]

    sh = json.load(open(b / f"shafts/{site}.geojson", encoding="utf-8"))["features"]
    lon = np.array([f["geometry"]["coordinates"][0] for f in sh], float)
    lat = np.array([f["geometry"]["coordinates"][1] for f in sh], float)
    e, n = geo.to_utm(lon, lat)
    S = np.column_stack([e, n])
    states = np.array([f["properties"].get("state", "unknown") for f in sh])
    return feats, S, states


def prep_chains(feats, S, states, tol=config.CHAIN_TOL_M):
    """Per-chain axis, measured spacing, and the state mix of its own shafts.

    Both endpoints are oriented south-to-north so ``pos_along`` has one meaning
    across chains: 0 is the downstream end (where a qanat reaches the surface
    and depth tends to zero), 1 is the upstream end (deepest). Chains that run
    nearly east-west cannot support that reading; ``ew_frac`` reports it rather
    than hiding it.
    """
    out = []
    for f in feats:
        p = f["properties"]
        c = np.asarray(f["geometry"]["coordinates"], float)
        e, n = geo.to_utm(c[:, 0], c[:, 1])
        a = np.array([e[0], n[0]])
        b = np.array([e[1], n[1]])
        if b[1] < a[1]:                       # make a the southern end
            a, b = b, a
        v = b - a
        L = float(np.hypot(*v))
        if L < 1e-6:
            continue
        u = v / L
        nrm = np.array([-u[1], u[0]])

        rel = S - a
        s = rel @ u
        d = np.abs(rel @ nrm)
        m = (d <= tol) & (s >= -1.0) & (s <= L + 1.0)
        n_m = int(m.sum())
        act = float((states[m] == "active").sum() / n_m) if n_m else float("nan")

        spacing = float(p.get("gap_spacing_m") or 0.0)
        if spacing <= 0:
            spacing = config.SHAFT_SPACING_DEFAULT_M

        out.append({
            "cid": p.get("cid"), "a": a, "b": b, "u": u, "L": L,
            "spacing": spacing, "n_shaft": int(p.get("n_shaft") or n_m),
            "grade": p.get("grade"), "active_share": act,
            "ew_frac": float(abs(v[0]) / L),
        })
    return out


def _plan_rows(site, scan="data/out/scan_{site}.csv"):
    with open(scan.format(site=site), encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def crossing_rows(site, chains, scan=None, hw=HW, b_road=B_ROAD):
    """Every crossing of every scan plan, plus a per-plan roll-up.

    The plan geometry is rebuilt exactly as ``replay_plans`` does it, so the
    event count per plan must equal the ``lines`` column the scan table already
    published. That equality is checked in ``run`` rather than assumed.
    """
    scan = scan or f"data/out/scan_{site}.csv"
    rows = _plan_rows(site, scan)
    P = np.vstack([np.vstack([c["a"], c["b"]]) for c in chains])
    center = np.array([np.median(P[:, 0]), np.median(P[:, 1])])

    ev, pl = [], []
    for r in rows:
        brg = float(r["bearing_deg"])
        off = float(r["offset_m"])
        length = float(r["length_km"]) * 1000.0
        pts = profile.plan_from(center, brg, length, off)
        pu = _unit(pts[1] - pts[0])
        pid = f"b{brg:05.1f}_o{off:+06.0f}"

        cavity = 0.0
        for c in chains:
            clipped = cases._clip_length(c["a"], c["b"], pts[0], pts[1], hw)
            cavity += min(clipped, c["L"])

        hits_total = 0
        csum = 0.0
        nsum = 0.0
        alphas = []
        for c in chains:
            hits = geo.polyline_crossings(pts, np.vstack([c["a"], c["b"]]))
            if not hits:
                continue
            hits_total += 1
            h = hits[0]
            alpha = _fold_deg(pu, c["u"])
            s = max(0.0, min(1.0, h["chainage_b"] / c["L"]))
            if alpha < 1e-9:
                alpha = 1e-9
            # Length of this chain inside the formation itself. For a chain
            # that crosses the whole formation this equals B/sin(a); taking it
            # from the same clipper as the corridor length keeps the two
            # consistent when the crossing sits near a chain end, where the
            # analytic value would overstate the loaded length.
            coupling = cases._clip_length(c["a"], c["b"], pts[0], pts[1], b_road / 2.0)
            coupling = min(coupling, c["L"])
            in_corr = cases._clip_length(c["a"], c["b"], pts[0], pts[1], hw)
            in_corr = min(in_corr, c["L"])
            sp = c["spacing"]
            ev.append({
                "site": site, "plan": pid, "bearing_deg": f"{brg:.1f}",
                "offset_m": f"{off:.0f}", "cid": c["cid"],
                "alpha_deg": round(alpha, 2), "n_seg": len(hits),
                "coupling_m": round(coupling, 1),
                "in_corridor_m": round(in_corr, 1),
                "spacing_m": round(sp, 1),
                "n_open_road": round(coupling / sp, 2),
                "n_open_corr": round(in_corr / sp, 2),
                "chain_len_m": round(c["L"], 1),
                "n_shaft": c["n_shaft"], "grade": c["grade"],
                "active_share": (round(c["active_share"], 3)
                                 if c["active_share"] == c["active_share"] else ""),
                "pos_along": round(s, 3), "ew_frac": round(c["ew_frac"], 3),
                "x_m": round(h["x"], 1), "y_m": round(h["y"], 1),
            })
            csum += coupling
            nsum += coupling / sp
            alphas.append(alpha)

        pl.append({
            "site": site, "plan": pid, "bearing_deg": f"{brg:.1f}",
            "offset_m": f"{off:.0f}", "rel_deg": r.get("rel_deg", ""),
            "n_events": hits_total,
            "coupling_sum": round(csum, 1),
            "n_open_road_sum": round(nsum, 2),
            "in_corridor_sum": round(cavity, 1),
            "alpha_min": round(min(alphas), 2) if alphas else "",
            "alpha_med": round(float(np.median(alphas)), 2) if alphas else "",
            "_scan_lines": int(r["lines"]),
            "_scan_cavity_km": float(r["cavity_km"]),
            "_cavity_m": cavity,
        })
    return ev, pl


def bin_summary(ev, pl, bins=((0, 15), (15, 30), (30, 45), (45, 60), (60, 75), (75, 90.01))):
    """Events and plans tabulated by the observed crossing angle.

    Binned by the crossing angle itself, which is the opposite of T3: there the
    bin is the plan-to-fan angle, because the question was which alignments an
    operator can choose. Here the question is what a single crossing looks like,
    so the observed angle is the right axis.

    ``rate`` divides the count by the integral of sin(a) over the bin, because
    that is the density a uniform bearing sweep produces on its own. Without it
    the 0-15 deg bin looks 7.7 times rarer than 75-90 deg purely as geometry.
    """
    out = []
    for site in sorted({e["site"] for e in ev}):
        sub = [e for e in ev if e["site"] == site]
        plans = [p for p in pl if p["site"] == site]
        lc_tot = sum(e["coupling_m"] for e in sub)
        no_tot = sum(e["n_open_road"] for e in sub)
        for lo, hi in bins:
            m = [e for e in sub if lo <= e["alpha_deg"] < hi]
            if not m:
                continue
            ids = {e["plan"] for e in m}
            si = math.cos(math.radians(lo)) - math.cos(math.radians(min(hi, 90.0)))
            out.append({
                "site": site, "alpha_bin": f"{lo}-{min(hi, 90):.0f}",
                "n_events": len(m),
                "n_plans": len(ids),
                "share_events": round(len(m) / len(sub), 4),
                "rate_per_sin": round(len(m) / si) if si > 0 else "",
                "coupling_med_m": round(float(np.median([e["coupling_m"] for e in m])), 1),
                "in_corridor_med_m": round(float(np.median([e["in_corridor_m"] for e in m])), 1),
                "n_open_road_med": round(float(np.median([e["n_open_road"] for e in m])), 2),
                "spacing_med_m": round(float(np.median([e["spacing_m"] for e in m])), 1),
                "coupling_share": round(sum(e["coupling_m"] for e in m) / lc_tot, 4),
                "n_open_share": round(sum(e["n_open_road"] for e in m) / no_tot, 4),
                "pos_along_med": round(float(np.median([e["pos_along"] for e in m])), 3),
                "ew_frac_med": round(float(np.median([e["ew_frac"] for e in m])), 3),
            })
        # plan level
        out.append({
            "site": site, "alpha_bin": "all",
            "n_events": len(sub), "n_plans": len(plans),
            "share_events": 1.0, "rate_per_sin": "",
            "coupling_med_m": round(float(np.median([e["coupling_m"] for e in sub])), 1),
            "in_corridor_med_m": round(float(np.median([e["in_corridor_m"] for e in sub])), 1),
            "n_open_road_med": round(float(np.median([e["n_open_road"] for e in sub])), 2),
            "spacing_med_m": round(float(np.median([e["spacing_m"] for e in sub])), 1),
            "coupling_share": 1.0, "n_open_share": 1.0,
            "pos_along_med": round(float(np.median([e["pos_along"] for e in sub])), 3),
            "ew_frac_med": round(float(np.median([e["ew_frac"] for e in sub])), 3),
        })
    return out


def _write(rows, path, cols):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return path


ROAD_FILE = {
    "studyA": "data/roads/aoi_b.geojson",   # registry: aoi_b contains studyA_core
    "study2": "data/roads/aoi_a.geojson",   # registry: aoi_a contains study2
}

ROAD_COLS = [
    "site", "road", "ref", "highway", "width_m", "seg", "cid",
    "alpha_deg", "coupling_m", "spacing_m", "n_open_road", "pos_along",
    "seg_len_m", "x_m", "y_m",
]


def road_events(site, chains, roads=None):
    """Existing OSM roads against the chains, one row per crossing.

    Kept separate from ``crossing_rows`` on purpose: the scan grid samples the
    alignments an operator *could* build, the road network is what *was* built.
    They are different populations and must never be pooled into one angle
    distribution.
    """
    rp = Path(roads or ROAD_FILE[site])
    if not rp.exists():
        return []
    feats = json.load(open(rp, encoding="utf-8"))["features"]
    out = []
    for f in feats:
        c = np.asarray(f["geometry"]["coordinates"], float)
        if c.ndim != 2 or len(c) < 2:
            continue
        p = f["properties"]
        e, n = geo.to_utm(c[:, 0], c[:, 1])
        w = float(p.get("width_m") or config.ROAD_WIDTH.get(
            str(p.get("highway", "")).lower(), ("class4", 6.5))[1])
        for i in range(len(e) - 1):
            seg = np.array([[e[i], n[i]], [e[i + 1], n[i + 1]]])
            seg_len = float(np.hypot(*(seg[1] - seg[0])))
            if seg_len < 1.0:
                continue
            u = _unit(seg[1] - seg[0])
            for k in chains:
                hits = geo.polyline_crossings(seg, np.vstack([k["a"], k["b"]]))
                for h in hits:
                    a = max(_fold_deg(u, k["u"]), 1e-9)
                    coup = min(cases._clip_length(k["a"], k["b"], seg[0], seg[1], w / 2.0),
                               k["L"])
                    out.append({
                        "site": site,
                        "road": p.get("osm_id", ""), "ref": p.get("ref", ""),
                        "highway": p.get("highway", ""), "width_m": round(w, 1),
                        "seg": i, "cid": k["cid"],
                        "alpha_deg": round(a, 2),
                        "coupling_m": round(coup, 1),
                        "spacing_m": round(k["spacing"], 1),
                        "n_open_road": round(coup / k["spacing"], 2),
                        "pos_along": round(max(0.0, min(1.0, h["chainage_b"] / k["L"])), 3),
                        "seg_len_m": round(seg_len, 1),
                        "x_m": round(h["x"], 1), "y_m": round(h["y"], 1),
                    })
    return out



def run(sites=("studyA", "study2"), outdir="data/out", hw=HW, b_road=B_ROAD,
        base="data", verbose=True):
    """Write events, plan roll-up and the angle-bin summary; assert the ties."""
    od = Path(outdir)
    all_ev, all_pl, all_sum, all_road = [], [], [], []
    report = []

    for site in sites:
        feats, S, states = load_site(site, base)
        chains = prep_chains(feats, S, states)
        ev, pl = crossing_rows(site, chains, scan=f"{base}/out/scan_{site}.csv",
                               hw=hw, b_road=b_road)

        # 1. events per plan == the lines column the scan already published
        for p in pl:
            assert p["n_events"] == p["_scan_lines"], (
                f"{site} {p['plan']}: {p['n_events']} crossings != scan {p['_scan_lines']}")
        # 2. sum of capped corridor lengths == the scan's cavity total.
        # The scan stores cavity to 3 decimals of a kilometre, i.e. to a metre,
        # so the comparison is made at the stored precision rather than tighter:
        # asserting sub-metre agreement against a metre-grained column would be
        # testing the CSV's rounding, not the geometry.
        for p in pl:
            assert round(p["_cavity_m"] / 1000.0, 3) == p["_scan_cavity_km"], (
                f"{site} {p['plan']}: cavity {p['_cavity_m']/1000:.4f} km "
                f"!= scan {p['_scan_cavity_km']:.4f} km")
        # 3. a crossing angle of exactly 0 means collinear, which cannot cross
        for e in ev:
            assert 0.0 < e["alpha_deg"] <= 90.0, f"{site} {e['plan']} {e['cid']}: alpha {e['alpha_deg']}"
        # 4. corridor length is bounded by chain length, and coupling by corridor
        for e in ev:
            assert e["in_corridor_m"] <= e["chain_len_m"] + 0.05
            assert e["coupling_m"] <= e["in_corridor_m"] + 0.05

        _write(ev, od / f"events_{site}.csv", COLS)
        _write([{k: v for k, v in p.items() if not k.startswith("_")} for p in pl],
               od / f"plans_{site}.csv", PLAN_COLS)

        al = np.array([e["alpha_deg"] for e in ev])
        rdev = road_events(site, chains)
        all_road += rdev
        report.append({
            "site": site, "plans": len(pl), "events": len(ev),
            "alpha_med": float(np.median(al)),
            "alpha_p10": float(np.percentile(al, 10)),
            "alpha_p90": float(np.percentile(al, 90)),
            "coupling_med": float(np.median([e["coupling_m"] for e in ev])),
            "n_open_road_med": float(np.median([e["n_open_road"] for e in ev])),
            "road_hits": len(rdev),
        })
        all_ev += ev
        all_pl += pl
        all_sum += bin_summary(ev, pl)

    _write(all_sum, od / "angle_summary.csv", list(all_sum[0].keys()))
    if all_road:
        _write(all_road, od / "events_roads.csv", ROAD_COLS)

    if verbose:
        print(f"{'site':8s} {'plans':>6s} {'events':>7s} {'a_med':>6s} "
              f"{'a_p10':>6s} {'a_p90':>6s} {'Lc_med':>7s} {'nopen':>6s} {'roads':>6s}")
        for r in report:
            print(f"{r['site']:8s} {r['plans']:6d} {r['events']:7d} "
                  f"{r['alpha_med']:6.1f} {r['alpha_p10']:6.1f} {r['alpha_p90']:6.1f} "
                  f"{r['coupling_med']:7.1f} {r['n_open_road_med']:6.2f} {r['road_hits']:6d}")
        print(f"\nwritten {od}/events_*.csv, plans_*.csv, angle_summary.csv")
    return all_ev, all_pl, all_sum
