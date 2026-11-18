"""Chain rebuild: shafts -> polyline axes (RANSAC over candidate directions).

This is core increment 0 in the pipeline: without an axis there is no crossing
angle, and without an angle the whole geometric argument has nothing to stand
on. The algorithm is deliberately conservative - a chain that cannot be
supported by a clean line is dropped rather than guessed.

Speed note: instead of the textbook "sample a pair, count inliers, repeat until
nothing is left" loop (which costs ~M*N per accepted chain), every candidate
pair is scored once in a single chunked pass, then clusters are taken greedily
by support. On 4k shafts this is seconds rather than minutes.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from . import config, geo


def _score_pairs(P, pairs, tol, pad, chunk=300):
    """For each candidate pair, count inliers and their span."""
    A = P[pairs[:, 0]]
    B = P[pairs[:, 1]]
    V = B - A
    D = np.hypot(V[:, 0], V[:, 1])
    U = V / D[:, None]

    n = len(P)
    best_cnt = np.zeros(len(pairs), dtype=np.int32)
    best_span = np.zeros(len(pairs), dtype=np.float64)
    masks = []

    for start in range(0, len(pairs), chunk):
        stop = min(start + chunk, len(pairs))
        u = U[start:stop]
        a = A[start:stop]
        d = D[start:stop]
        rel = P[None, :, :] - a[:, None, :]
        s = rel[:, :, 0] * u[:, 0:1] + rel[:, :, 1] * u[:, 1:2]
        lat = np.abs(-rel[:, :, 0] * u[:, 1:2] + rel[:, :, 1] * u[:, 0:1])
        inl = (lat <= tol) & (s >= -pad) & (s <= d[:, None] + pad)
        cnt = inl.sum(axis=1)
        s_masked = np.where(inl, s, np.nan)
        span = np.nanmax(s_masked, axis=1) - np.nanmin(s_masked, axis=1)
        best_cnt[start:stop] = cnt
        best_span[start:stop] = np.nan_to_num(span)
        masks.append(inl)
    return best_cnt, best_span, np.vstack(masks)


def _candidate_pairs(P, tree, gap_min, gap_max, bins=(25.0, 70.0, 140.0, 250.0), per_bin=2):
    """Multi-scale candidate pairs: nearest few neighbours in each distance band.

    A pure kNN list is not enough - in dense clusters the 8 nearest points are
    all within 30 m, so a pair that spans a 200 m detection gap is never even
    proposed, and the chain gets cut at every hole. Sampling one or two points
    per distance band keeps the pair count linear while still covering the long
    gaps.
    """
    bins = [b for b in bins if gap_min < b <= max(gap_max, gap_min + 1)]
    if not bins:
        bins = [gap_max]
    neighbours = tree.query_ball_point(P, max(max(bins), gap_max))

    pairs = set()
    for i, nb in enumerate(neighbours):
        buckets = {b: [] for b in bins}
        for j in nb:
            if j <= i:
                continue
            d = float(np.hypot(*(P[j] - P[i])))
            if d < gap_min or d > gap_max:
                continue
            for b in bins:
                if d <= b:
                    buckets[b].append((d, j))
                    break
        for b in bins:
            for d, j in sorted(buckets[b])[:per_bin]:
                pairs.add((i, int(j)))
    return np.array(sorted(pairs), dtype=np.int64)


def _merge_collinear(chains, P, tol=6.0, angle_tol=12.0, gap_tol=70.0, offset_tol=18.0):
    """Join chain fragments that lie on the same axis."""
    merged = True
    while merged:
        merged = False
        for i in range(len(chains)):
            for j in range(i + 1, len(chains)):
                a, b = chains[i], chains[j]
                ua = (a["p1"] - a["p0"]) / max(np.hypot(*(a["p1"] - a["p0"])), 1e-9)
                ub = (b["p1"] - b["p0"]) / max(np.hypot(*(b["p1"] - b["p0"])), 1e-9)
                ang = np.degrees(np.arccos(min(abs(float(ua @ ub)), 1.0)))
                if ang > angle_tol:
                    continue
                # Distance from each fragment to the other's *axis*, not to its
                # endpoints. Endpoint-to-endpoint misses the common case where a
                # short fragment lies wholly inside a longer one: its ends are
                # hundreds of metres from the long chain's ends even though the
                # two describe the same tunnel.
                def _p2seg(p, c):
                    return float(geo.point_segment_distance(p[0], p[1], c["p0"][0], c["p0"][1],
                                                            c["p1"][0], c["p1"][1]))
                dmin = min(_p2seg(a["p0"], b), _p2seg(a["p1"], b),
                           _p2seg(b["p0"], a), _p2seg(b["p1"], a))
                if dmin > gap_tol:
                    continue
                # lateral offset between the two axes
                ends_b = np.array([b["p0"], b["p1"]])
                off = np.min(np.abs((ends_b - a["p0"]) @ np.array([-ua[1], ua[0]])))
                if off > offset_tol:
                    continue
                idx = np.union1d(a["idx"], b["idx"])
                fused = _refine(P, idx, tol=max(a["rms"], b["rms"], 4.0))
                if fused is None:
                    continue
                chains[i] = fused
                chains.pop(j)
                merged = True
                break
            if merged:
                break
    return chains


def audit_overlaps(chains, angle_tol=12.0, offset_tol=18.0, frac=0.3):
    """Find pairs of chains that describe the same tunnel.

    ``_merge_collinear`` merges greedily and can stop short: two fragments that
    are only discovered as a pair after a third one has already absorbed one of
    them stay separate. A duplicate inflates the corridor table twice over - two
    crossings where there is one, and twice the cavity length in the same place -
    so the count is worth auditing on real data rather than trusting the merge
    loop to have converged.

    The test is deliberately stricter than the merge predicate: both endpoints
    of the shorter chain must sit within ``offset_tol`` of the longer one's axis
    *and* the projected overlap must cover more than ``frac`` of the shorter
    chain. Taking a minimum over endpoint distances (the older test) passes any
    short fragment that merely touches a long one at one end.
    """
    out = []
    for i in range(len(chains)):
        for j in range(i + 1, len(chains)):
            a, b = chains[i], chains[j]
            pa = np.asarray([a["p0"], a["p1"]])
            pb = np.asarray([b["p0"], b["p1"]])
            va, vb = pa[1] - pa[0], pb[1] - pb[0]
            la, lb = float(np.hypot(*va)), float(np.hypot(*vb))
            if la < 1.0 or lb < 1.0:
                continue
            ua, ub = va / la, vb / lb
            ang = float(np.degrees(np.arccos(min(abs(float(ua @ ub)), 1.0))))
            if ang > angle_tol:
                continue
            nrm = np.array([-ua[1], ua[0]])
            off = float(np.max(np.abs((pb - pa[0]) @ nrm)))
            if off > offset_tol:
                continue
            s = (pb - pa[0]) @ ua
            overlap = min(la, float(s.max())) - max(0.0, float(s.min()))
            if overlap <= 0 or overlap / min(la, lb) <= frac:
                continue
            out.append({"i": i, "j": j, "angle_deg": round(ang, 1),
                        "offset_m": round(off, 1), "overlap_m": round(overlap, 0),
                        "len_i": round(la, 0), "len_j": round(lb, 0)})
    return out


def rebuild(xy, tol=None, min_pts=None, min_span=None,
            gap_min=25.0, gap_max=250.0, pad=40.0, keep_ratio=0.3,
            per_bin=2, merge=True):
    """Fit chain axes to shaft coordinates (metres, already projected).

    Returns a list of dicts with inlier indices, endpoints and fit quality.
    """
    tol = config.CHAIN_TOL_M if tol is None else tol
    min_pts = config.CHAIN_MIN_PTS if min_pts is None else min_pts
    min_span = config.CHAIN_MIN_SPAN_M if min_span is None else min_span

    P = np.asarray(xy, float)
    if len(P) < min_pts:
        return []

    tree = cKDTree(P)
    pairs = _candidate_pairs(P, tree, gap_min, gap_max, per_bin=per_bin)
    if len(pairs) == 0:
        return []

    cnt, span, masks = _score_pairs(P, pairs, tol, pad)
    ok = (cnt >= min_pts) & (span >= min_span)
    if not ok.any():
        return []

    order = np.argsort(-cnt[ok])
    pairs_ok = pairs[ok][order]
    masks_ok = masks[ok][order]

    used = np.zeros(len(P), bool)
    found = []
    for k in range(len(pairs_ok)):
        inl = masks_ok[k]
        fresh = inl & ~used
        if fresh.sum() < keep_ratio * inl.sum() or fresh.sum() < min_pts:
            continue
        idx = np.where(fresh)[0]
        c = _refine(P, idx, tol)
        if c is not None:
            found.append(c)
        used |= inl

    if merge:
        found = _merge_collinear(found, P, tol=max(tol, 4.0))

    # The caller asked for a minimum span; enforce it on the fitted axis, not on
    # the pre-fit score. Trimming to the 1st/99th percentile can shorten a small
    # cluster to a few metres, so 14 chains shorter than --min-span used to slip
    # through and were reported as chains.
    found = [c for c in found if c["length"] >= min_span]

    # The greedy merge can still leave a pair describing one tunnel, and a
    # duplicate is counted twice in every corridor statistic. audit_overlaps
    # already detects them; leaving the detection and the data disagreeing is the
    # worst of both, so fold them here until the audit comes back empty.
    found = merge_duplicates(found, xy, tol=tol)

    found.sort(key=lambda c: -c["n"])
    return found


def merge_duplicates(found, xy, tol=None, **kw):
    """Fold chains that the audit says describe the same tunnel.

    Merging re-fits the union of both shaft sets rather than welding two axes, so
    the surviving chain has an honest residual. Repeated until clean, because
    merging one pair can expose another (a third fragment may have absorbed a
    shaft that made the first pair separable).
    """
    tol = config.CHAIN_TOL_M if tol is None else tol
    out = list(found)
    for _ in range(len(out)):
        dups = audit_overlaps(out, **kw)
        if not dups:
            break
        d = dups[0]
        keep, drop = ((d["i"], d["j"]) if d["len_i"] >= d["len_j"]
                      else (d["j"], d["i"]))
        idx = np.union1d(out[keep]["idx"], out[drop]["idx"])
        refit = _refine(xy, idx, tol)
        if refit is not None:
            out[keep] = refit
        out.pop(drop)
    return out


def _refine(P, idx, tol):
    pts = P[idx]
    centre = pts.mean(axis=0)
    q = pts - centre
    # principal direction via SVD; a chain is a thin cloud
    _, _, vt = np.linalg.svd(q, full_matrices=False)
    u = vt[0]
    s = q @ u
    lat = np.abs(-q[:, 0] * u[1] + q[:, 1] * u[0])
    rms = float(np.sqrt((lat ** 2).mean()))
    if rms > tol * 1.5:
        return None
    # trim to the 2nd/98th percentile so a single stray shaft cannot stretch the axis
    lo, hi = np.percentile(s, [1, 99])
    p0 = centre + u * lo
    p1 = centre + u * hi
    spacing = float(np.median(np.diff(np.sort(s)))) if len(s) > 2 else 0.0
    # coverage: observed shafts over the number the measured spacing predicts.
    # Without it a chain with eight points spread over 900 m grades as A even
    # though six of its shafts are missing, which is exactly the case a caller
    # needs to be warned about.
    expect = (float(hi - lo) / spacing + 1.0) if spacing > 0 else float(len(idx))
    cover = float(min(1.0, len(idx) / max(expect, 1.0)))
    return {
        "idx": idx,
        "p0": p0,
        "p1": p1,
        "n": int(len(idx)),
        "rms": rms,
        "length": float(hi - lo),
        "spacing": spacing,
        "cover": cover,
    }


def extend(p0, p1, extra: float):
    """Lengthen a chain at both ends (tunnel continues past the last shaft)."""
    v = p1 - p0
    L = np.hypot(*v)
    if L == 0:
        return p0, p1
    u = v / L
    return p0 - u * extra, p1 + u * extra


def gap_stats(P, idx, p0, p1):
    """Longest shaft-free stretch inside a chain, and the spacing that judges it.

    The residual cannot tell a tunnel passing under a village from two different
    rows fused by the lateral tolerance - both look like a line with shafts near
    it. The gap can: a fused chain always carries a long hole where the second
    row crosses the axis, while a single row thins out gradually. On the six
    hand-checked chains the rejected one had three holes of 105/182/136 m and
    every accepted one stayed under 76 m.
    """
    pts = P[np.asarray(idx, int)]
    v = np.asarray(p1, float) - np.asarray(p0, float)
    L = float(np.hypot(*v))
    if len(pts) < 3 or L <= 0:
        return {"gap_max": 0.0, "spacing": 0.0, "ratio": 0.0, "n": int(len(pts))}
    u = v / L
    s = np.sort((pts - np.asarray(p0, float)) @ u)
    g = np.diff(s)
    med = float(np.median(g))
    return {"gap_max": float(g.max()), "spacing": med,
            "ratio": float(g.max() / med) if med > 0 else 0.0, "n": int(len(s))}


def grade_chain(rms, n, cover, gap_max, spacing):
    """Grade a chain, then demote it one step if it spans a long shaft-free hole.

    Two separate questions: how well do the shafts fit a line (residual,
    coverage), and is the thing actually one continuous row (gap). The old
    grading only asked the first, so a chain that fused two rows across a 180 m
    hole could still score A. Splitting the two keeps the fit quality readable
    while making the continuity defect visible.
    """
    if rms <= 2.5 and n >= 8 and cover >= 0.55:
        base = "A"
    elif rms <= 4.0 and cover >= 0.30:
        base = "B"
    else:
        base = "C"
    frag = gap_max > max(config.CHAIN_GAP_FLAG_M,
                        config.CHAIN_GAP_RATIO * max(spacing, 1e-9))
    demoted = {"A": "B", "B": "C", "C": "C"}[base] if frag else base
    return {"grade": demoted, "base": base, "fragmented": bool(frag)}
