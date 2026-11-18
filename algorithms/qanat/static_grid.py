"""Offline routing-grid functions extracted from qanat/static_build.py."""

from __future__ import annotations

import base64

import numpy as np

CELL = 12.0
PAD = 320.0
RADII = (15.0, 8.0)


def _utm(bbox):
    """bbox corners in UTM 45N, padded - the routing grid of one site."""
    from . import geo
    w, s, e, n = bbox
    lon = np.array([w, w, e, e])
    lat = np.array([s, n, s, n])
    x, y = geo.to_utm(lon, lat, zone=45, northern=True)
    x0, y0 = float(x.min()) - PAD, float(y.min()) - PAD
    x1, y1 = float(x.max()) + PAD, float(y.max()) + PAD
    nx = int((x1 - x0) / CELL) + 1
    ny = int((y1 - y0) / CELL) + 1
    return x0, y0, nx, ny


def _raster(rows, origin, nx, ny):
    mask = np.zeros((ny, nx), bool)
    for r in rows:
        (ax, ay), (bx, by) = r["p0"], r["p1"]
        k = int(max(2, np.hypot(bx - ax, by - ay) / CELL * 2))
        xs = np.linspace(ax, bx, k)
        ys = np.linspace(ay, by, k)
        ix = np.clip(((xs - origin[0]) / CELL).astype(int), 0, nx - 1)
        iy = np.clip(((ys - origin[1]) / CELL).astype(int), 0, ny - 1)
        mask[iy, ix] = True
    return mask


def _bits(arr):
    """Pack a boolean grid into a base64 string, one bit per cell."""
    return base64.b64encode(np.packbits(arr.reshape(-1).astype(np.uint8)).tobytes()).decode()


def _field(mask):
    from scipy.ndimage import distance_transform_edt
    f = np.exp(-(distance_transform_edt(~mask) * CELL / 18.0) ** 2)
    return base64.b64encode((f * 255.0).round().astype(np.uint8).tobytes()).decode()


def build_site(site: dict) -> dict:
    origin = _utm(site["bbox"])
    x0, y0, nx, ny = origin
    rows = site["rows"]
    mask = _raster(rows, (x0, y0), nx, ny)
    from scipy.ndimage import distance_transform_edt
    dist = distance_transform_edt(~mask) * CELL
    out = {k: site[k] for k in ("id", "name", "name_en", "note", "note_en", "bbox",
                                "zone", "img", "size", "shafts", "rows", "roads",
                                "crossings", "stats")}
    out["grid"] = {"x0": x0, "y0": y0, "cell": CELL, "nx": nx, "ny": ny}
    out["field"] = _field(mask)
    out["block"] = [_bits(dist < r) for r in RADII]
    return out
