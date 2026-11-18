"""Run Q-RITE algorithms on invented geometries and a synthetic brightness profile."""

import json

import numpy as np

from qanat import cases, chains, geo, iran_refine, platform_crossings
from qanat import profile, rows, static_grid, typology, wtransect


def main():
    easting, northing = geo.to_utm(89.2, 42.9, zone=45, northern=True)
    origin = np.array([easting, northing], dtype=float)

    def lonlat(points):
        points = np.asarray(points, dtype=float)
        lon, lat = geo.to_lonlat(points[:, 0], points[:, 1], zone=45, northern=True)
        return np.column_stack((lon, lat)).tolist()

    shafts = origin + np.column_stack((np.arange(0.0, 251.0, 25.0), np.zeros(11)))
    rebuilt = chains.rebuild(shafts)
    traced = rows.extract(shafts)
    assert len(rebuilt) == len(traced) == 1

    core = np.delete(shafts, 4, axis=0)
    merged, _ = iran_refine.merge(core, lonlat(shafts[[4]]), zone=45)
    assert len(merged) == len(shafts)
    grade = chains.grade_chain(0.0, len(shafts), 1.0, 25.0, 25.0)["grade"]
    assert grade == "A"

    axis = shafts[[0, -1]]
    row = {"p0": axis[0].tolist(), "p1": axis[1].tolist(), "spacing": 25.0}
    road = origin + np.array([[110.0, -50.0], [110.0, 50.0]])
    ways = [{"pts": lonlat(road), "w": 26.0, "cls": "example", "name": "Synthetic road"}]
    contact, = platform_crossings.assess([row], ways, zone=45)
    assert contact["L"] == 26.0 and contact["mouths"] == 1.04

    parallel = origin + np.array([[60.0, 8.0], [160.0, 8.0]])
    footprint = cases.evaluate_plan(parallel, [axis], shafts, hw=13.0)
    assert footprint["lines"] == 0 and footprint["cavity_km"] == 0.1

    bins, _ = profile.bins(road, [axis], shafts, step=50.0, hw=13.0)
    contacts = typology.contacts(road, [axis], hw=13.0)
    assert sum(b["cavity_m"] for b in bins) == 26.0 and contacts[0]["type"] == 1

    corners = lonlat(origin + np.array([[-100.0, -100.0], [350.0, 100.0]]))
    site = dict.fromkeys(("name", "name_en", "note", "note_en", "img", "size",
                          "roads", "stats", "shafts", "crossings"))
    site.update(id="synthetic", bbox=[*corners[0], *corners[1]], zone=45, rows=[row])
    grid = static_grid.build_site(site)
    assert grid["field"] and len(grid["block"]) == 2

    offsets = np.arange(-26.0, 26.01, 0.25)
    brightness = np.where(np.abs(offsets) <= 4.0, 55.0, 130.0)

    def synthetic_profile(centre, tangent, along=16.0):
        return offsets, brightness

    case = dict(case="example", kind="crossing", site="synthetic", road="example",
                highway="test", class_width_m=8.0, alpha_deg=90.0,
                c=road.mean(axis=0), t=np.array([0.0, 1.0]))
    width = wtransect.measure(case, profile_fn=synthetic_profile)
    assert width["accepted"] == 1 and width["measured_m"] == 8.0

    print(json.dumps({
        "rebuilt_rows": len(rebuilt),
        "traced_rows": len(traced),
        "recovered_shafts": len(merged) - len(core),
        "row_grade": grade,
        "crossing_angle_deg": contact["alpha"],
        "crossing_overlap_m": contact["L"],
        "estimated_shaft_intervals": contact["mouths"],
        "parallel_crossings": footprint["lines"],
        "parallel_footprint_overlap_m": footprint["cavity_km"] * 1000.0,
        "profile_overlap_m": sum(b["cavity_m"] for b in bins),
        "routing_grid": [grid["grid"]["nx"], grid["grid"]["ny"]],
        "transect_width_m": width["measured_m"],
    }, indent=2))


if __name__ == "__main__":
    main()
