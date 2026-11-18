"""Offline crossing calculations extracted from qanat/live.py."""

import math

import numpy as np

from . import geo


def _alpha_at(a_xy, ca, b_xy, cb, half=50.0):
    """Acute angle between two polylines at a contact, from local tangents."""
    b1 = geo.tangent_near(np.asarray(a_xy, float), ca, half)
    b2 = geo.tangent_near(np.asarray(b_xy, float), cb, half)
    ang = abs(geo.angle_between_deg(b1, b2)) % 180.0
    return 180.0 - ang if ang > 90.0 else ang


def assess(rows, ways, zone, width_m=None):
    """Every row x road contact, with L, mouths and the angle behind them."""
    ways_utm = []
    for w in ways:
        e, n = geo.to_utm(np.array([p[0] for p in w["pts"]]),
                          np.array([p[1] for p in w["pts"]]),
                          zone=zone, northern=True)
        ways_utm.append(dict(w, xy=np.column_stack([e, n])))

    out = []
    for ri, r in enumerate(rows):
        a = np.array([r["p0"], r["p1"]], float)
        for wi, w in enumerate(ways_utm):
            b = np.asarray(w["xy"], float)
            for c in geo.polyline_crossings(a, b):
                alpha = max(_alpha_at(a, c["chainage_a"], b, c["chainage_b"]), 5.0)
                W = width_m or w["w"]
                L = min(W / math.sin(math.radians(alpha)), 400.0)
                lon, lat = geo.to_lonlat(c["x"], c["y"], zone=zone, northern=True)
                out.append({"row": ri, "way": wi, "cls": w["cls"],
                            "name": w["name"], "w": W,
                            "alpha": round(alpha, 1), "L": round(L, 1),
                            "mouths": round(L / (r["spacing"] or 1.0), 2),
                            "x": round(c["x"], 1), "y": round(c["y"], 1),
                            "lon": float(lon), "lat": float(lat)})
    return out
