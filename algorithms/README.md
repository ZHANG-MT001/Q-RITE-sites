# Q-RITE Algorithms

[中文说明](README_zh.md) · [Q-RITE map](../index.html) · [Map technical notes](../TECHNICAL.md)

This folder contains the project-specific code used to reconstruct qanat shaft rows, compute their geometric overlap with roads, and compare candidate alignments. The five-window map ships with precomputed shaft rows and crossings. Its routing engine runs in the browser when a user picks endpoints.

## Algorithm index

| Algorithm | Entry point | What it computes |
| --- | --- | --- |
| 1. Road–row crossing geometry | [`platform_crossings.assess`](qanat/platform_crossings.py), [`angles.road_events`](qanat/angles.py) | Intersects mapped road centrelines and inferred qanat axes. Reports intersection points, acute angles, overlap lengths and spacing-based estimates of shaft intervals. |
| 2. Avoidance cost grid and route generation | [`static_grid.build_site`](qanat/static_grid.py), [`planRouteLocal`](browser/router.js), [`route.build_density`](qanat/route.py) | Rasterises qanat axes into a distance-based cost field and two exclusion masks for the viewer. The browser uses A* to generate alternatives at several avoidance weights. The research module provides a separate Python cost field and route search. |
| 3. Recommendation under a detour limit | [`pickBest`](browser/detour.js) | Sets a length limit relative to the straight reference. Among eligible alternatives, prefers lower estimated overlap and breaks ties by shorter route length. |
| 4. Shaft-row reconstruction | [`chains.rebuild`](qanat/chains.py), [`rows.extract`](qanat/rows.py) | Links detected shaft points with multi-scale pair selection and collinear merging, or traces rows along their local direction. |
| 5. Corridor re-detection and row grading | [`iran_refine.collect` and `merge`](qanat/iran_refine.py), [`chains.grade_chain`](qanat/chains.py) | Searches for missed shafts along a fitted row and beyond its ends, then filters candidates by lateral offset and spacing. Grades rows by fit, shaft count, coverage and long gaps. |
| 6. Full road-footprint overlap | [`cases.evaluate_plan`](qanat/cases.py), [`footprint`](qanat/footprint.py) | Clips mapped axes to a finite road rectangle. The paper's analysis compares complete overlap with overlap counted only at centreline crossings. |
| 7. Along-route profiles and contact types | [`profile.bins`](qanat/profile.py), [`typology.contacts`](qanat/typology.py) | Summarises exposure along a route by chainage and distinguishes transverse, oblique, parallel-in-buffer and nearby parallel contacts. |
| 8. Road width from image transects | [`wtransect.measure`](qanat/wtransect.py), [`iran_crossing`](qanat/iran_crossing.py) | Samples brightness across a mapped road, estimates its edges at half the trough depth, and checks contrast, centring and width stability. |

## Run the example

From this directory, with Python 3.10 or later:

```bash
python -m pip install -r requirements.txt
python -B demo.py
```

The demo uses synthetic coordinates and a synthetic brightness profile. It reconstructs one row, recovers one missing shaft, calculates a 26 m overlap at a right-angle crossing, finds 100 m of footprint overlap on a parallel route with zero centreline crossings, and recovers an 8 m road width. It also builds the two routing masks. Once the Python dependencies are installed, this example works offline.

## Reuse the code

- **Existing crossings.** Call `platform_crossings.assess(rows, ways, zone=45)` for the Turpan UTM zone. Each row supplies projected endpoint arrays `p0`, `p1` and median `spacing` in metres. Each road supplies `pts` as `[longitude, latitude]` pairs, width `w` in metres, `cls` and `name`.
- **Browser routing.** `browser/router.js` is the viewer's route engine. Call `window.planRouteLocal(waypoints, site)` with at least two `[longitude, latitude]` pairs and a site payload in the format used by the viewer's `sites/data.js`. The payload contains the row geometry, `grid`, `field` and packed `block` masks. `browser/detour.js` exposes `pickBest(candidates, budget)` with `0.10` for a 10% cap and `1` for the viewer's unlimited option. The route list must include the engine's straight reference.
- **Research geometry.** Pass projected, metre-based NumPy coordinate arrays to `chains.rebuild`, `rows.extract`, `cases.evaluate_plan`, `profile.bins` and `typology.contacts`. For a straight alignment, the road rectangle has half-width `hw`. Use `hw=13.0` for a 26 m formation.
- **Research data.** The original batch routines read the study's `data/` tree. Re-running the paper's tables or measuring widths on study imagery requires the matching study inputs and tile cache. The optional second-pass detector additionally requires Ultralytics and compatible YOLO weights supplied by the caller. `wtransect.measure` also accepts a `profile_fn` for callers who already have brightness profiles.

## How to interpret the outputs

The viewer scores centreline crossings using a width-over-sine estimate, with safeguards for very small angles. Its planned-road example uses a 26 m formation width. Use the full-footprint calculation for overlaps alongside the centreline or near the road ends. A spacing-based estimate such as 1.04 represents equivalent shaft intervals. Geometry is inferred from mapped axes and supports screening for route comparison and preliminary investigation.

Road centrelines used by the map come from OpenStreetMap. The shaft detector uses YOLOv8s. A*, Liang–Barsky clipping and Chaikin smoothing are standard methods used by the implementation. This folder contains source code and synthetic examples. The imagery and trained weights are supplied separately in the corresponding project workflows.

## Source notes

Core Python modules are copied from the Q-RITE research toolkit. `platform_crossings.py` and `static_grid.py` extract the relevant functions from `live.py` and `static_build.py`. `browser/router.js` is copied from the delivered viewer. `browser/detour.js` extracts the viewer's recommendation rule with the budget passed as an argument. The corridor re-detection module uses a relative default weight path, and the transect module accepts a caller-supplied profile for the demo.
