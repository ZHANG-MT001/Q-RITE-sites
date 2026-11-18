"""The single exchange format: WGS84 GeoJSON with three geometry types only.

Everything crossing a module boundary goes through here, so field names and
value types are enforced in one place. If a module needs a new attribute, add
it to SCHEMA below first - never ad hoc on a feature.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import config, geo

SCHEMA = {
    "aoi": {
        "geometry": "Polygon",
        "required": {"name": str},
        "desc": "study-area rectangle; the single entry point of every computation",
    },
    "shaft": {
        "geometry": "Point",
        "required": {"conf": float, "state": str, "grade": str, "est": bool},
        "desc": "one detected shaft; est=True means extrapolated, not observed",
    },
    "chain": {
        "geometry": "LineString",
        "required": {"cid": str, "n_shaft": int, "grade": str, "resid_m": float},
        "desc": "reconstructed shaft chain (qanat tunnel axis)",
    },
    "road": {
        "geometry": "LineString",
        "required": {"osm_id": str, "highway": str, "road_class": str, "width_m": float},
        "desc": "existing road centreline with class and formation width",
    },
    "alignment": {
        "geometry": "LineString",
        "required": {"aid": str, "facility": str, "length_m": float},
        "desc": "planned alignment: highway / railway / hsr / tunnel",
    },
    "grade": {
        "geometry": "LineString",
        "required": {"seg_id": str, "i_norm": float, "level": str},
        "desc": "a profile segment carrying normalised interference intensity",
    },
}

FACILITIES = ("highway", "railway", "hsr", "tunnel")
LEVELS = ("low", "medium", "high")
STATES = ("active", "inactive", "unknown")
GRADES = ("A", "B", "C")


class SchemaError(ValueError):
    pass


def empty_collection() -> dict:
    return {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": f"urn:ogc:def:crs:{config.CRS_EXCHANGE.replace(':', '::')}"}},
        "features": [],
    }


def feature(kind: str, coords, **props) -> dict:
    if kind not in SCHEMA:
        raise SchemaError(f"unknown kind {kind!r}")
    return {
        "type": "Feature",
        "geometry": {"type": SCHEMA[kind]["geometry"], "coordinates": coords},
        "properties": {"kind": kind, **props},
    }


def _check_types(kind: str, props: dict) -> None:
    for key, typ in SCHEMA[kind]["required"].items():
        if key not in props:
            raise SchemaError(f"{kind}: missing required property {key!r}")
        val = props[key]
        if typ is float and isinstance(val, int):
            continue
        if not isinstance(val, typ):
            raise SchemaError(f"{kind}.{key}: expected {typ.__name__}, got {type(val).__name__}")
    if kind == "shaft" and props["state"] not in STATES:
        raise SchemaError(f"shaft.state must be one of {STATES}, got {props['state']!r}")
    if kind == "alignment" and props["facility"] not in FACILITIES:
        raise SchemaError(f"alignment.facility must be one of {FACILITIES}, got {props['facility']!r}")
    if kind == "grade" and props["level"] not in LEVELS:
        raise SchemaError(f"grade.level must be one of {LEVELS}, got {props['level']!r}")
    if kind in ("shaft", "chain") and props.get("grade") not in GRADES:
        raise SchemaError(f"{kind}.grade must be one of {GRADES}, got {props.get('grade')!r}")


def _iter_coords(geom: dict):
    t = geom["type"]
    if t == "Point":
        yield geom["coordinates"]
    elif t in ("LineString", "MultiPoint"):
        yield from geom["coordinates"]
    elif t in ("Polygon", "MultiLineString"):
        for ring in geom["coordinates"]:
            yield from ring
    else:
        raise SchemaError(f"geometry type {t!r} not allowed; use Point / LineString / Polygon")


def validate(collection: dict, strict: bool = True) -> dict:
    if collection.get("type") != "FeatureCollection":
        raise SchemaError("root must be a FeatureCollection")
    features = collection.get("features")
    if not isinstance(features, list):
        raise SchemaError("features must be a list")

    counter: dict[str, int] = {}
    for i, feat in enumerate(features):
        if feat.get("type") != "Feature":
            raise SchemaError(f"feature[{i}] is not a Feature")
        props = feat.get("properties") or {}
        kind = props.get("kind")
        if kind not in SCHEMA:
            raise SchemaError(f"feature[{i}] has unknown kind {kind!r}")
        geom = feat.get("geometry") or {}
        if geom.get("type") != SCHEMA[kind]["geometry"]:
            raise SchemaError(
                f"feature[{i}] {kind}: geometry must be {SCHEMA[kind]['geometry']}, got {geom.get('type')}"
            )
        _check_types(kind, props)
        for lon, lat in _iter_coords(geom):
            if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
                raise SchemaError(f"feature[{i}] coordinate out of WGS84 range: {lon}, {lat}")
        counter[kind] = counter.get(kind, 0) + 1

    if strict:
        crs_name = ((collection.get("crs") or {}).get("properties") or {}).get("name", "")
        if crs_name and "4326" not in crs_name:
            raise SchemaError(f"exchange CRS must be EPSG:4326, got {crs_name!r}")
    return counter


def read(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    validate(data)
    return data


def _round_coords(obj, precision: int):
    if isinstance(obj, (int, float)):
        return round(float(obj), precision)
    if isinstance(obj, list):
        return [_round_coords(o, precision) for o in obj]
    return obj


def write(path: str | Path, collection: dict, precision: int = 7) -> Path:
    validate(collection)
    out = json.loads(json.dumps(collection))
    for feat in out["features"]:
        geom = feat["geometry"]
        geom["coordinates"] = _round_coords(geom["coordinates"], precision)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return p


def to_utm(feat: dict) -> np.ndarray:
    """Project one feature's coordinates to EPSG:32645 metres.

    This is the ONLY place a coordinate transformation happens. Downstream
    code never calls it again - it receives metres.
    """
    geom = feat["geometry"]
    t = geom["type"]
    if t == "Point":
        lon, lat = geom["coordinates"]
        e, n = geo.to_utm(lon, lat)
        return np.array([[e, n]])
    if t == "LineString":
        arr = np.asarray(geom["coordinates"], float)
        e, n = geo.to_utm(arr[:, 0], arr[:, 1])
        return np.column_stack([e, n])
    if t == "Polygon":
        arr = np.asarray(geom["coordinates"][0], float)
        e, n = geo.to_utm(arr[:, 0], arr[:, 1])
        return np.column_stack([e, n])
    raise SchemaError(f"to_utm does not handle {t}")


def from_utm(xy: np.ndarray, kind: str, **props) -> dict:
    xy = np.asarray(xy, float)
    lon, lat = geo.to_lonlat(xy[:, 0], xy[:, 1])
    coords = list(zip(np.round(lon, 7).tolist(), np.round(lat, 7).tolist()))
    if kind == "shaft":
        coords = coords[0]
    return feature(kind, coords, **props)
