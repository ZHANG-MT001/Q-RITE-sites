"""Geodesy and planar geometry without shapely / pyproj.

Only numpy is required, so the kernel runs anywhere (sandbox, lab PC, AutoDL
node) regardless of GDAL availability. Forward/inverse Transverse Mercator
follow Snyder (1987) series; agreement with PROJ is at the millimetre level
inside a 3 degree band around the central meridian, which covers the basin.
"""

from __future__ import annotations

import math

import numpy as np

from . import config

# WGS84 ellipsoid
_A = 6378137.0
_F = 1.0 / 298.257223563
_E2 = _F * (2.0 - _F)
_EP2 = _E2 / (1.0 - _E2)
_K0 = 0.9996
_FALSE_EASTING = 500000.0
_FALSE_NORTHING = 10000000.0


def central_meridian(zone: int) -> float:
    return zone * 6.0 - 183.0


def _meridian_arc(phi: np.ndarray) -> np.ndarray:
    e2, e4, e6 = _E2, _E2 ** 2, _E2 ** 3
    return _A * (
        (1 - e2 / 4 - 3 * e4 / 64 - 5 * e6 / 256) * phi
        - (3 * e2 / 8 + 3 * e4 / 32 + 45 * e6 / 1024) * np.sin(2 * phi)
        + (15 * e4 / 256 + 45 * e6 / 1024) * np.sin(4 * phi)
        - (35 * e6 / 3072) * np.sin(6 * phi)
    )


def to_utm(lon, lat, zone: int | None = None, northern: bool | None = None):
    """WGS84 degrees -> UTM easting/northing in metres.

    Accepts scalars or arrays; returns floats for scalar input.
    """
    zone = config.UTM_ZONE if zone is None else zone
    northern = config.UTM_NORTHERN if northern is None else northern

    scalar = np.isscalar(lon)
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)

    phi = np.radians(lat)
    dlam = np.radians(lon - central_meridian(zone))
    sin_p, cos_p, tan_p = np.sin(phi), np.cos(phi), np.tan(phi)

    n = _A / np.sqrt(1 - _E2 * sin_p ** 2)
    t = tan_p ** 2
    c = _EP2 * cos_p ** 2
    a = cos_p * dlam
    m = _meridian_arc(phi)

    easting = _K0 * n * (
        a
        + (1 - t + c) * a ** 3 / 6
        + (5 - 18 * t + t ** 2 + 72 * c - 58 * _EP2) * a ** 5 / 120
    ) + _FALSE_EASTING

    northing = _K0 * (
        m
        + n * tan_p * (
            a ** 2 / 2
            + (5 - t + 9 * c + 4 * c ** 2) * a ** 4 / 24
            + (61 - 58 * t + t ** 2 + 600 * c - 330 * _EP2) * a ** 6 / 720
        )
    )
    if not northern:
        northing = northing + _FALSE_NORTHING

    if scalar:
        return float(easting), float(northing)
    return easting, northing


def to_lonlat(easting, northing, zone: int | None = None, northern: bool | None = None):
    """UTM metres -> WGS84 lon/lat degrees."""
    zone = config.UTM_ZONE if zone is None else zone
    northern = config.UTM_NORTHERN if northern is None else northern

    scalar = np.isscalar(easting)
    x = np.asarray(easting, dtype=float) - _FALSE_EASTING
    y = np.asarray(northing, dtype=float)
    if not northern:
        y = y - _FALSE_NORTHING

    m = y / _K0
    mu = m / (_A * (1 - _E2 / 4 - 3 * _E2 ** 2 / 64 - 5 * _E2 ** 3 / 256))
    e1 = (1 - math.sqrt(1 - _E2)) / (1 + math.sqrt(1 - _E2))
    j1 = 3 * e1 / 2 - 27 * e1 ** 3 / 32
    j2 = 21 * e1 ** 2 / 16 - 55 * e1 ** 4 / 32
    j3 = 151 * e1 ** 3 / 96
    j4 = 1097 * e1 ** 4 / 512
    phi1 = mu + j1 * np.sin(2 * mu) + j2 * np.sin(4 * mu) + j3 * np.sin(6 * mu) + j4 * np.sin(8 * mu)

    sin1, cos1, tan1 = np.sin(phi1), np.cos(phi1), np.tan(phi1)
    c1 = _EP2 * cos1 ** 2
    t1 = tan1 ** 2
    n1 = _A / np.sqrt(1 - _E2 * sin1 ** 2)
    r1 = _A * (1 - _E2) / (1 - _E2 * sin1 ** 2) ** 1.5
    d = x / (_K0 * n1)

    lat = phi1 - (n1 * tan1 / r1) * (
        d ** 2 / 2
        - (5 + 3 * t1 + 10 * c1 - 4 * c1 ** 2 - 9 * _EP2) * d ** 4 / 24
        + (61 + 90 * t1 + 298 * c1 + 45 * t1 ** 2 - 252 * _EP2 - 3 * c1 ** 2) * d ** 6 / 720
    )
    lon = np.radians(central_meridian(zone)) + (
        d
        - (1 + 2 * t1 + c1) * d ** 3 / 6
        + (5 - 2 * c1 + 28 * t1 - 3 * c1 ** 2 + 8 * _EP2 + 24 * t1 ** 2) * d ** 5 / 120
    ) / cos1

    if scalar:
        return float(np.degrees(lon)), float(np.degrees(lat))
    return np.degrees(lon), np.degrees(lat)


# --- planar helpers (metres, already projected) -------------------------

def polyline_length(xy: np.ndarray) -> float:
    xy = np.asarray(xy, dtype=float)
    if len(xy) < 2:
        return 0.0
    return float(np.sum(np.hypot(*np.diff(xy, axis=0).T)))


def resample_line(xy: np.ndarray, step: float) -> np.ndarray:
    """Resample a polyline at (approximately) constant spacing `step` metres.

    Returns an (n, 3) array: x, y, cumulative chainage at each sample.
    """
    xy = np.asarray(xy, dtype=float)
    seg = np.hypot(*np.diff(xy, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total == 0:
        return np.array([[xy[0, 0], xy[0, 1], 0.0]])
    n = max(int(math.ceil(total / step)), 1)
    targets = np.linspace(0.0, total, n + 1)
    x = np.interp(targets, s, xy[:, 0])
    y = np.interp(targets, s, xy[:, 1])
    return np.column_stack([x, y, targets])


def bearing_deg(x1, y1, x2, y2) -> float:
    """Planar bearing in degrees, 0 = +x (east), increasing counter-clockwise."""
    return float(np.degrees(np.arctan2(np.asarray(y2) - np.asarray(y1),
                                       np.asarray(x2) - np.asarray(x1))))


def angle_between_deg(b1: float, b2: float) -> float:
    """Acute angle between two directions, 0-90 degrees.

    This is the crossing angle alpha used everywhere: no orientation, so a
    chain crossing a road at 30 deg and at 150 deg are the same hazard.
    """
    d = abs((b1 - b2 + 180.0) % 360.0 - 180.0)
    return 90.0 - abs(d - 90.0)


def polyline_crossings(a_xy: np.ndarray, b_xy: np.ndarray):
    """All proper intersections between two polylines.

    Returns a list of dicts: chainage on a, chainage on b, and x/y in metres.
    Segment counts here are in the hundreds, so the double loop is fine and
    far easier to audit than a vectorised sweep.
    """
    a = np.asarray(a_xy, float)
    b = np.asarray(b_xy, float)
    if len(a) < 2 or len(b) < 2:
        return []

    seg_a = np.diff(a, axis=0)
    seg_b = np.diff(b, axis=0)
    len_a = np.hypot(*seg_a.T)
    len_b = np.hypot(*seg_b.T)
    s_a = np.concatenate([[0.0], np.cumsum(len_a)])
    s_b = np.concatenate([[0.0], np.cumsum(len_b)])

    out = []
    for i in range(len(seg_a)):
        p, r = a[i], seg_a[i]
        for j in range(len(seg_b)):
            q, s = b[j], seg_b[j]
            denom = r[0] * s[1] - r[1] * s[0]
            if abs(denom) < 1e-12:
                continue
            qp = q - p
            t = (qp[0] * s[1] - qp[1] * s[0]) / denom
            u = (qp[0] * r[1] - qp[1] * r[0]) / denom
            if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
                out.append({
                    "chainage_a": float(s_a[i] + t * len_a[i]),
                    "chainage_b": float(s_b[j] + u * len_b[j]),
                    "x": float(p[0] + t * r[0]),
                    "y": float(p[1] + t * r[1]),
                })
    return out


def tangent_near(xy: np.ndarray, chainage: float, half_window: float) -> float:
    """Local bearing at `chainage`, fitted over +-half_window metres.

    Uses the two window endpoints rather than the whole polyline: a road is
    curved, and the global chord would bias alpha systematically.
    """
    xy = np.asarray(xy, float)
    seg = np.hypot(*np.diff(xy, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    lo = max(chainage - half_window, 0.0)
    hi = min(chainage + half_window, s[-1])
    if hi <= lo:
        return float("nan")
    x0 = np.interp(lo, s, xy[:, 0])
    y0 = np.interp(lo, s, xy[:, 1])
    x1 = np.interp(hi, s, xy[:, 0])
    y1 = np.interp(hi, s, xy[:, 1])
    return bearing_deg(x0, y0, x1, y1)


def point_segment_distance(px, py, ax, ay, bx, by):
    """Perpendicular distance from points to segment ab, vectorised."""
    px, py = np.asarray(px, float), np.asarray(py, float)
    abx, aby = bx - ax, by - ay
    denom = abx ** 2 + aby ** 2
    if denom == 0:
        return np.hypot(px - ax, py - ay)
    t = np.clip(((px - ax) * abx + (py - ay) * aby) / denom, 0.0, 1.0)
    return np.hypot(px - (ax + t * abx), py - (ay + t * aby))


def segments_intersect(p1, p2, q1, q2):
    """Vectorised segment-segment intersection test. Returns bool array."""
    def cross(o, a, b):
        return (a[..., 0] - o[..., 0]) * (b[..., 1] - o[..., 1]) - (a[..., 1] - o[..., 1]) * (b[..., 0] - o[..., 0])

    p1, p2, q1, q2 = (np.asarray(v, float) for v in (p1, p2, q1, q2))
    d1 = cross(q1, q2, p1)
    d2 = cross(q1, q2, p2)
    d3 = cross(p1, p2, q1)
    d4 = cross(p1, p2, q2)
    return ((d1 * d2) < 0) & ((d3 * d4) < 0)


def interior_angles_deg(xy: np.ndarray) -> np.ndarray:
    """Turn angle at each interior vertex, degrees (for smoothness checks)."""
    xy = np.asarray(xy, float)
    if len(xy) < 3:
        return np.empty(0)
    v1 = xy[1:-1] - xy[:-2]
    v2 = xy[2:] - xy[1:-1]
    num = np.sum(v1 * v2, axis=1)
    den = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
    cos = np.clip(num / np.where(den == 0, 1, den), -1.0, 1.0)
    return np.degrees(np.arccos(cos))
