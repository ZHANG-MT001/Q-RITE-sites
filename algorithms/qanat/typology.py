"""Geometric typology of qanat-alignment interference.

The corridor table returns four scalars - length, crossings, shafts, cavity. Two
of them are ambiguous in a way that matters for design: a plan scoring the same
Lq can be (a) fifteen transverse crossings of a 100 m span each, or (b) one 1.5 km
run alongside a tunnel. Those are not the same problem. Crossing puts a short
span of void under the carriageway fifteen times; running alongside puts the road
on top of one continuous void for a kilometre and a half.

So exposure is decomposed rather than summed. Every (alignment segment, chain
segment) pair inside the corridor is classified by two quantities that are
measured, not assumed - whether the two actually cross, and the local angle
between them:

  I   正穿 transverse   crosses, local angle >= 60 deg
  II  斜穿 oblique      crosses, local angle <  60 deg
  III 并行贴线 parallel, inside the statutory buffer
  IV  并行近接 parallel, inside the analysis corridor but outside the buffer

The buffer is the building-control distance of the Regulation on the Protection
of Highway Safety (State Council Order 593, Article 11): 30 m expressway, 20
first class, 15 second, 10 third, 5 fourth. Type III is the case that regulation
is actually about; Type IV is not a violation but is still inside the influence
zone the mechanics layer will need.

Why a typology and not a threshold: the thresholds that would rank these types
belong to the mechanics layer (burial ratio, state weight, settlement limits) and
that layer is not calibrated yet. What is available now is geometry - and the
geometry already separates cases that the scalar Lq merges. Calling the split by
its proper name keeps the claim honest: this classifies, it does not score.

Consistency requirement: the four Lq shares must sum to the corridor table's
cavity, because they are the same clipping operation partitioned. That is checked
in selftest, not left to inspection.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from . import config, figs, geo
from .cases import _alpha, _clip_length

TRANSVERSE_MIN = 60.0
PARALLEL_MAX = 30.0

TYPES = (
    (1, "正穿", "transverse"),
    (2, "斜穿", "oblique"),
    (3, "并行贴线", "parallel, in buffer"),
    (4, "并行近接", "parallel, near"),
)


def load_alignments(path):
    """Read every alignment feature as (label, xy_in_metres).

    Label comes from the ``lam`` property when the file was written by
    ``route --sweep``, otherwise from ``aid``. Shared by the command and the
    figure builder so the two can never disagree about which line is which.
    """
    from . import io_geo

    out = []
    for f in io_geo.read(path)["features"]:
        if f["properties"].get("kind") != "alignment":
            continue
        lam = f["properties"].get("lam")
        label = f"λ={lam:g}" if lam is not None else str(f["properties"].get("aid", "plan"))
        out.append((label, io_geo.to_utm(f)))
    return out


def facility_of(path, default="highway"):
    """Facility recorded on the first alignment in the file."""
    from . import io_geo

    for f in io_geo.read(path)["features"]:
        if f["properties"].get("kind") == "alignment":
            return str(f["properties"].get("facility", default))
    return default


def pick(feats, wanted):
    """Nearest available alignment to each wanted label value."""
    picked, seen = [], set()
    for w in wanted:
        best = min(feats, key=lambda f: abs(_label_value(f[0]) - w))
        if best[0] not in seen:
            seen.add(best[0])
            picked.append(best)
    return picked


def _label_value(label, default=0.0):
    try:
        return float(str(label).split("=")[-1])
    except (TypeError, ValueError):
        return default


def _chainage(align, p):
    """Distance along the alignment of the point nearest to p."""
    best = (np.inf, 0.0)
    acc = 0.0
    for i in range(len(align) - 1):
        a, b = align[i], align[i + 1]
        seg = float(np.hypot(*(b - a)))
        d = float(geo.point_segment_distance(p[0], p[1], a[0], a[1], b[0], b[1]))
        if d < best[0]:
            proj = float((np.asarray(p) - a) @ (b - a)) / max(seg ** 2, 1e-12)
            best = (d, acc + min(max(proj, 0.0), 1.0) * seg)
        acc += seg
    return best[1]


def contacts(align, chains, hw=100.0, buffer=30.0):
    """One row per (alignment segment, chain segment) pair inside the corridor.

    Clipping is the same Liang-Barsky routine the corridor table uses, and the
    per-chain total is capped the same way, so the shares add up to Lq exactly.
    """
    align = np.asarray(align, float)
    rows = []
    for ci, ch in enumerate(chains):
        ch = np.asarray(ch, float)
        for j in range(len(ch) - 1):
            c0, c1 = ch[j], ch[j + 1]
            for i in range(len(align) - 1):
                a, b = align[i], align[i + 1]
                clipped = _clip_length(c0, c1, a, b, hw)
                if clipped <= 1e-9:
                    continue
                across = bool(geo.segments_intersect(a[None, :], b[None, :],
                                                     c0[None, :], c1[None, :])[0])
                alpha = _alpha(a, b, c0, c1)
                if across:
                    kind = 1 if alpha >= TRANSVERSE_MIN else 2
                    dist = 0.0
                else:
                    dist = min(
                        float(geo.point_segment_distance(c0[0], c0[1], a[0], a[1], b[0], b[1])),
                        float(geo.point_segment_distance(c1[0], c1[1], a[0], a[1], b[0], b[1])),
                        float(geo.point_segment_distance(a[0], a[1], c0[0], c0[1], c1[0], c1[1])),
                        float(geo.point_segment_distance(b[0], b[1], c0[0], c0[1], c1[0], c1[1])),
                    )
                    # A chain inside the corridor that never crosses: if the pair is
                    # more oblique than PARALLEL_MAX it is an approach, not a run -
                    # still type IV, but the angle is reported so it can be seen.
                    kind = 3 if dist <= buffer else 4
                rows.append({
                    "chain": ci, "pair": (i, j), "type": kind,
                    "alpha_deg": round(alpha, 1), "dist_m": round(dist, 1),
                    "clipped_m": clipped, "s_m": round(_chainage(align, 0.5 * (c0 + c1)), 1),
                })

    # cap per chain, exactly as the corridor table does
    for ci in {r["chain"] for r in rows}:
        tot = sum(r["clipped_m"] for r in rows if r["chain"] == ci)
        cap = float(geo.polyline_length(np.asarray(chains[ci], float)))
        if tot > cap > 0:
            k = cap / tot
            for r in rows:
                if r["chain"] == ci:
                    r["clipped_m"] *= k
    return rows


def buffer_length(align, chains, buffer=30.0, step=2.0):
    """Metres of the alignment sitting inside the statutory buffer of any chain.

    Returns None when the facility has no statutory distance (a tunnel): the
    quantity does not exist rather than being zero, and reporting 0.0 would read
    as compliance.
    """
    if buffer is None:
        return None
    align = np.asarray(align, float)
    seg = np.diff(align, axis=0)
    L = np.hypot(seg[:, 0], seg[:, 1])
    total = float(L.sum())
    if total < step:
        return 0.0
    n = max(2, int(total / step))
    t = np.linspace(0.0, total, n)
    cum = np.concatenate([[0.0], np.cumsum(L)])
    pts = np.column_stack([np.interp(t, cum, align[:, 0]), np.interp(t, cum, align[:, 1])])
    d = np.full(n, np.inf)
    for ch in chains:
        ch = np.asarray(ch, float)
        for j in range(len(ch) - 1):
            d = np.minimum(d, geo.point_segment_distance(
                pts[:, 0], pts[:, 1], ch[j, 0], ch[j, 1], ch[j + 1, 0], ch[j + 1, 1]))
    frac = float((d <= buffer).mean())
    return frac * total


def classify(align, chains, shafts=None, hw=100.0, buffer=None,
             facility="highway", zone="other"):
    """Per-plan type counts, Lq shares and buffer occupancy.

    ``buffer`` defaults to the statutory distance of ``facility`` (config.buffer_for).
    Pass it explicitly only to run a comparison; a facility with no statutory
    distance leaves Type III empty and La_buffer undefined, which is the honest
    answer for a tunnel rather than a made-up number.
    """
    align = np.asarray(align, float)
    if buffer is None:
        buffer = config.buffer_for(facility, zone=zone)
    rows = contacts(align, chains, hw=hw, buffer=(buffer if buffer is not None else 0.0))
    L = float(geo.polyline_length(align))
    lb = buffer_length(align, chains, buffer)
    out = {
        "length_km": round(L / 1000.0, 3),
        "n_contact": len(rows),
        "Lq_km": round(sum(r["clipped_m"] for r in rows) / 1000.0, 4),
        "La_buffer_km": None if lb is None else round(lb / 1000.0, 4),
        "hw": hw, "buffer": buffer, "facility": facility,
    }
    # per-chain dominant type for the plan view
    dom = {}
    for r in rows:
        if r["chain"] not in dom or r["clipped_m"] > dom[r["chain"]]["clipped_m"]:
            dom[r["chain"]] = r
    out["dominant"] = dom
    for k, _, _ in TYPES:
        sub = [r for r in rows if r["type"] == k]
        out[f"n_{k}"] = len({r["chain"] for r in sub})
        out[f"Lq_{k}"] = round(sum(r["clipped_m"] for r in sub) / 1000.0, 4)
        al = [r["alpha_deg"] for r in sub if r["type"] in (1, 2)]
        out[f"alpha_{k}"] = round(float(np.median(al)), 1) if al else None
    return out


def report(features, chains, hw=100.0, buffer=None, facility="highway", zone="other"):
    """features: list of (label, xy) alignments in metres."""
    rows = []
    for label, xy in features:
        st = classify(xy, chains, hw=hw, buffer=buffer, facility=facility, zone=zone)
        st["plan"] = label
        st.pop("dominant", None)
        rows.append(st)
    return rows


def write_csv(rows, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    keys = ["plan", "length_km", "n_contact"] + [
        f"{pre}_{k}" for k, _, _ in TYPES for pre in ("n", "Lq")
    ] + ["Lq_km", "La_buffer_km", "hw", "buffer", "alpha_1", "alpha_2"]
    keys = [k for k in dict.fromkeys(keys) if k in rows[0]]
    with p.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return p


# --- figure ---------------------------------------------------------------
TYPE_COLOUR = {1: "#A33B2D", 2: "#C98A18", 3: "#1F6F8B", 4: "#7C93A8"}


def _paint(xy, chains, dom, hw, step=20.0):
    """Densify the alignment and colour each piece by its nearest chain's type.

    Colour per *piece*, not per vertex-to-vertex segment: after simplification a
    segment can be 800 m long and crosses a dozen chains, so colouring whole
    segments would paint a crossing and a parallel run the same colour.
    """
    seg = np.diff(xy, axis=0)
    L = np.hypot(seg[:, 0], seg[:, 1])
    total = float(L.sum())
    n = max(2, int(total / step))
    cum = np.concatenate([[0.0], np.cumsum(L)])
    t = np.linspace(0.0, total, n)
    pts = np.column_stack([np.interp(t, cum, xy[:, 0]), np.interp(t, cum, xy[:, 1])])

    best_d = np.full(n, np.inf)
    best_t = np.zeros(n, int)
    for ci, r in dom.items():
        ch = np.asarray(chains[ci], float)
        d = np.full(n, np.inf)
        for j in range(len(ch) - 1):
            d = np.minimum(d, geo.point_segment_distance(
                pts[:, 0], pts[:, 1], ch[j, 0], ch[j, 1], ch[j + 1, 0], ch[j + 1, 1]))
        m = d < best_d
        best_d[m] = d[m]
        best_t[m] = r["type"]
    best_t = np.where(best_d <= hw, best_t, 0)
    return pts, best_t


def plot(features, chains, st_by_label, outdir="figs", name="fig8_typology",
         idx=None, lang="en"):
    figs.use_style(lang)
    zh = lang == "zh"
    fig, axes = figs.plt.subplots(1, 2, figsize=(figs.COL2, 86 / 25.4),
                                  gridspec_kw={"width_ratios": [1.1, 1.0]})

    ax = axes[0]
    for ch in chains:
        ax.plot(ch[:, 0] / 1000, ch[:, 1] / 1000, color="#C6C1B6", linewidth=0.7, zorder=1)
    label, xy = features[0 if idx is None else idx]
    st = st_by_label[label]
    pts, kind = _paint(xy, chains, st["dominant"], st["hw"])
    for k in range(len(pts) - 1):
        col = TYPE_COLOUR[kind[k]] if kind[k] else "#2C2C2A"
        ax.plot([pts[k, 0] / 1000, pts[k + 1, 0] / 1000],
                [pts[k, 1] / 1000, pts[k + 1, 1] / 1000],
                color=col, linewidth=3.0, solid_capstyle="butt", zorder=4)
    for k, zl, el in TYPES:
        ax.plot([], [], color=TYPE_COLOUR[k], linewidth=3.0, label=zl if zh else el)
    # zoom to the alignment, not to the whole fan: at fan scale a 60 m buffer
    # contact is two pixels wide and the typology is invisible
    pad = 450.0
    x0, x1 = (pts[:, 0].min() - pad) / 1000, (pts[:, 0].max() + pad) / 1000
    ymid = (pts[:, 1].min() + pts[:, 1].max()) / 2000.0
    ax.set_xlim(x0, x1)
    # An equal-aspect map shrinks its own box to fit the data, which left blank
    # bands above and below the panel. Widen the latitude window to the height
    # the box actually has instead, so the map fills the panel and 1 km east is
    # still the same length on the page as 1 km north.
    fig.canvas.draw()
    bb = ax.get_window_extent()
    yspan = (x1 - x0) * (bb.height / bb.width)
    ax.set_ylim(ymid - yspan / 2.0, ymid + yspan / 2.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("E (km, UTM 45N)")
    ax.set_ylabel("N (km, UTM 45N)")
    fac = st.get("facility", "")
    buf = st.get("buffer")
    tag = f"{label}, {fac}" + (f", {buf:g} m" if buf else "")
    figs.scalebar(ax, (ax.get_xlim()[1] - ax.get_xlim()[0]), "km", loc=(0.06, 0.05))
    # Under the axis, not on the map: the lower-right corner is where the
    # densest bundle of grey axes runs, and the white plate that made the key
    # readable also blanked four of them out.
    # legend() fills column-major, so a ncol=2 key of four types reads
    # "transverse | parallel in buffer" / "oblique | parallel near". Passing the
    # handles in the order 0,2,1,3 puts the two parallel levels on the second
    # row, which is the order the caption describes them in.
    _h, _l = ax.get_legend_handles_labels()
    _o = [0, 2, 1, 3]
    leg = ax.legend([_h[i] for i in _o], [_l[i] for i in _o], loc="upper left",
                    bbox_to_anchor=(0.0, -0.14), ncol=2, fontsize=6.2,
                    handlelength=1.4, frameon=False, borderaxespad=0.0,
                    columnspacing=1.4, labelspacing=0.3)
    figs.panel(ax, "a")

    ax = axes[1]
    labels = [r["plan"] for r in st_by_label.values()]
    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))
    for k, zl, el in TYPES:
        vals = np.array([r[f"Lq_{k}"] for r in st_by_label.values()], float)
        if vals.max() <= 1e-9:
            continue
        ax.bar(x, vals, bottom=bottom, width=0.62, linewidth=0,
               color=TYPE_COLOUR[k], label=zl if zh else el)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.4, rotation=14, ha="right")
    ax.set_ylabel("按型式分解的 Lq (km)" if zh else "Lq split by type (km)")
    figs.grid(ax, axis="y")
    figs.panel(ax, "b")

    fig.subplots_adjust(wspace=0.26)
    return figs.save(fig, outdir, name)
