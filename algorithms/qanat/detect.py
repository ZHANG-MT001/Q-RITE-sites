"""Shaft detection over imagery tiles, streamed block by block.

The field project builds one mosaic for a site and detects on it. That works
while the site is small and the machine has memory to spare: at 0.22 m/px a
5 x 3.5 km site is 24 000 x 16 000 px, about 1.1 GB as an RGB array, before the
detector makes its own copy. On an 8 GB laptop with a browser open it does not
fit, and a failed mosaic means the whole run is lost.

So this module never holds the site. It walks the area in 1280 px blocks with a
one-tile overlap, keeps each block only long enough to run the detector, and
accumulates points. Peak memory is one block, independent of area - a 20 km site
costs the same as a 2 km one, it just takes longer.

Geometry is exact rather than calibrated: every detection sits at a fractional
tile coordinate, and the tile grid is defined by a formula, so pixel to lon/lat
needs no control points, no ICP, no manual GSD. Duplicates across block edges are
removed in UTM metres, where a distance is a distance.

    python -m qanat detect --bbox 89.268,42.769,89.335,42.801 --out data/shafts/study2.geojson
"""

from __future__ import annotations

import csv
import io as _io
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

TILE = 256
BLANK = np.full((TILE, TILE, 3), 232, dtype=np.uint8)   # pale grey, like empty imagery

_TL = threading.local()


def _session():
    import requests

    s = getattr(_TL, "s", None)
    if s is None:
        s = requests.Session()
        s.trust_env = False
        s.headers["User-Agent"] = "Mozilla/5.0 (karez-research; academic use)"
        _TL.s = s
    return s


# --- tile maths ----------------------------------------------------------
def lon2tx(lon, z):
    return (float(lon) + 180.0) / 360.0 * (2 ** z)


def lat2ty(lat, z):
    r = math.radians(float(lat))
    return (1.0 - math.log(math.tan(r) + 1.0 / math.cos(r)) / math.pi) / 2.0 * (2 ** z)


def tx2lon(tx, z):
    return tx / (2 ** z) * 360.0 - 180.0


def ty2lat(ty, z):
    n = math.pi - 2.0 * math.pi * ty / (2 ** z)
    return math.degrees(math.atan(math.sinh(n)))


def quadkey(tx, ty, z):
    qk = []
    for i in range(z, 0, -1):
        d = 0
        m = 1 << (i - 1)
        if tx & m:
            d += 1
        if ty & m:
            d += 2
        qk.append(str(d))
    return "".join(qk)


def tile_range(bbox, z):
    """Inclusive tile range covering (lon0, lat0, lon1, lat1)."""
    lon0, lat0, lon1, lat1 = bbox
    x0, x1 = int(math.floor(lon2tx(lon0, z))), int(math.floor(lon2tx(lon1, z)))
    y0, y1 = int(math.floor(lat2ty(lat1, z))), int(math.floor(lat2ty(lat0, z)))
    return x0, y0, x1, y1


def gsd(z, lat):
    """Ground sample distance in metres per pixel at a latitude and zoom."""
    return 156543.03392 * math.cos(math.radians(lat)) / (2 ** z)


# --- imagery -------------------------------------------------------------
def fetch_tile(x, y, z, cache=None, retries=3):
    """One 256 px tile as an array; blank (and counted) if it will not come."""
    if cache is not None:
        cp = Path(cache) / str(z) / str(x) / f"{y}.jpg"
        if cp.exists():
            try:
                from PIL import Image
                return np.asarray(Image.open(cp).convert("RGB"))
            except Exception:
                cp.unlink(missing_ok=True)
    qk = quadkey(x, y, z)
    urls = [f"https://ecn.t{i}.tiles.virtualearth.net/tiles/a{qk}.jpeg?g=1" for i in range(4)]
    for a in range(retries):
        try:
            r = _session().get(urls[a % 4], timeout=25)
            if r.status_code == 200 and r.content[:2] == b"\xff\xd8":
                if cache is not None:
                    cp = Path(cache) / str(z) / str(x) / f"{y}.jpg"
                    cp.parent.mkdir(parents=True, exist_ok=True)
                    cp.write_bytes(r.content)
                from PIL import Image
                return np.asarray(Image.open(_io.BytesIO(r.content)).convert("RGB"))
        except Exception:
            pass
        time.sleep(0.6 + 0.5 * a)
    return None


def block(x0, y0, ntiles, z, cache=None, workers=8):
    """Assemble an ntiles x ntiles block whose top-left tile is (x0, y0)."""
    out = np.empty((ntiles * TILE, ntiles * TILE, 3), dtype=np.uint8)
    misses = 0
    coords = [(x0 + i, y0 + j) for j in range(ntiles) for i in range(ntiles)]

    def one(c):
        return c, fetch_tile(c[0], c[1], z, cache)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for (x, y), img in ex.map(one, coords):
            i, j = x - x0, y - y0
            if img is None:
                misses += 1
                img = BLANK
            out[j * TILE:(j + 1) * TILE, i * TILE:(i + 1) * TILE] = img[:TILE, :TILE]
    return out, misses


# --- detection -----------------------------------------------------------
def _detect_block(model, img, conf, imgsz, device):
    """(col, row, conf, class) for every box centre, in block pixel coordinates."""
    r = model.predict(img, imgsz=imgsz, conf=conf, iou=0.7, agnostic_nms=True,
                      verbose=False, device=device)[0]
    out = []
    if r.boxes is None or len(r.boxes) == 0:
        return out
    xy = r.boxes.xywh.cpu().numpy()
    cf = r.boxes.conf.cpu().numpy()
    cl = r.boxes.cls.cpu().numpy().astype(int)
    for (cx, cy, _w, _h), c, k in zip(xy, cf, cl):
        out.append((float(cx), float(cy), float(c), int(k)))
    return out


def dedupe(pts, tol=8.0):
    """Greedy removal of the same shaft seen twice, in UTM metres.

    Block edges are the only source of duplicates, and the overlap is a whole
    tile, so the same hole can be found twice with centres a few metres apart.
    8 m is well under the shaft spacing (~25 m here) and well over the centre
    drift from a clipped box, so it merges twins without eating neighbours.
    """
    from . import geo

    if not pts:
        return []
    e, n = geo.to_utm(np.array([p["lon"] for p in pts]), np.array([p["lat"] for p in pts]))
    order = sorted(range(len(pts)), key=lambda i: -pts[i]["conf"])
    keep, grid, cell = [], {}, tol
    for i in order:
        cx, cy = int(e[i] // cell), int(n[i] // cell)
        clash = False
        for gx in (cx - 1, cx, cx + 1):
            for gy in (cy - 1, cy, cy + 1):
                for j in grid.get((gx, gy), ()):
                    if (e[i] - e[j]) ** 2 + (n[i] - n[j]) ** 2 <= tol * tol:
                        clash = True
                        break
                if clash:
                    break
            if clash:
                break
        if not clash:
            grid.setdefault((cx, cy), []).append(i)
            keep.append(i)
    return [pts[i] for i in sorted(keep)]


def run(bbox, zoom=19, weights=None, conf=0.2, imgsz=1280, block_tiles=5,
        stride_tiles=4, device="cpu", cache=None, workers=8, limit=None,
        log=print):
    """Detect over a bbox. Returns (points, info).

    ``limit`` stops after that many blocks, for a smoke test.
    """
    if weights is None:
        raise ValueError("weights are required: pass the trained .pt file")
    from ultralytics import YOLO

    model = YOLO(str(weights))
    x0, y0, x1, y1 = tile_range(bbox, zoom)
    lat_c = 0.5 * (bbox[1] + bbox[3])
    step = max(1, int(stride_tiles))
    xs = list(range(x0, x1 + 1, step))
    ys = list(range(y0, y1 + 1, step))
    total = len(xs) * len(ys)
    if limit:
        total = min(total, limit)

    pts, misses, done, outside = [], 0, 0, 0
    t0 = time.time()
    for j, ty in enumerate(ys):
        for i, tx in enumerate(xs):
            if limit and done >= limit:
                break
            img, m = block(tx, ty, block_tiles, zoom, cache, workers)
            misses += m
            for cx, cy, cf, k in _detect_block(model, img, conf, imgsz, device):
                lon = tx2lon(tx + cx / TILE, zoom)
                lat = ty2lat(ty + cy / TILE, zoom)
                # A block covers 5 tiles but steps 4, so it overhangs the bbox by
                # up to a tile on every side. Without this clip the density is
                # computed over an area larger than the one reported, and any
                # comparison against another site inherits the error.
                if not (bbox[0] <= lon <= bbox[2] and bbox[1] <= lat <= bbox[3]):
                    outside += 1
                    continue
                pts.append({"lon": lon, "lat": lat, "conf": cf,
                            "state": "active" if k == 0 else "inactive",
                            "px": round(cx + (tx - x0) * TILE, 1),
                            "py": round(cy + (ty - y0) * TILE, 1)})
            done += 1
            if log and (done % 20 == 0 or done == total):
                el = time.time() - t0
                log(f"  {done}/{total} blocks  {el:.0f}s  "
                    f"~{el / done * (total - done):.0f}s left  raw {len(pts)}  misses {misses}")
        if limit and done >= limit:
            break

    pts = dedupe(pts)
    info = {"bbox": [float(v) for v in bbox], "zoom": zoom,
            "gsd_m": gsd(zoom, lat_c), "blocks": done, "tile_misses": misses,
            "conf": conf, "imgsz": imgsz, "n": len(pts), "outside": outside,
            "active": sum(1 for p in pts if p["state"] == "active"),
            "seconds": round(time.time() - t0, 1)}
    return pts, info


def write(points, out_json, out_csv=None, weights=None):
    """Write the exchange-format GeoJSON and, optionally, a pixel csv."""
    from . import io_geo

    col = io_geo.empty_collection()
    for p in points:
        col["features"].append(io_geo.feature(
            "shaft", [round(p["lon"], 7), round(p["lat"], 7)],
            conf=round(p["conf"], 3), state=p["state"], grade="B", est=False))
    io_geo.validate(col)
    jp = io_geo.write(Path(out_json), col)
    cp = None
    if out_csv:
        cp = Path(out_csv)
        cp.parent.mkdir(parents=True, exist_ok=True)
        with cp.open("w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            w.writerow(["id", "cls", "conf", "lon", "lat", "px", "py"])
            for i, p in enumerate(points, 1):
                w.writerow([i, p["state"], round(p["conf"], 3), round(p["lon"], 7),
                            round(p["lat"], 7), p["px"], p["py"]])
    return jp, cp
