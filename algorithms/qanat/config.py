"""Project constants: CRS, AOI registry, engineering thresholds.

Every number that later appears in a paper table lives here, so that
changing it changes every figure at once.
"""

# --- coordinate contract ------------------------------------------------
CRS_EXCHANGE = "EPSG:4326"     # GeoJSON on disk, always lon/lat degrees
CRS_WORK = "EPSG:32645"        # UTM zone 45N, all metric computation
UTM_ZONE = 45
UTM_NORTHERN = True            # Turpan basin is ~42-43 N
SOURCE_CRS_HINT = "EPSG:3857"  # Bing / Google tile grids

# --- processing defaults ------------------------------------------------
SEGMENT_LEN_M = 50.0           # profile sampling along an alignment
LOCAL_TANGENT_HALF_M = 50.0    # half window for local tangent at a crossing
SHAFT_SPACING_DEFAULT_M = 40.0 # fallback when local spacing cannot be measured
SHAFT_SPACING_RANGE_M = (10.0, 70.0)

# --- chain rebuild ------------------------------------------------------
CHAIN_TOL_M = 6.0          # lateral tolerance of a shaft to the fitted axis
# 4, not 5: the published Turpan-wide screening keeps clusters of >= 3 shafts,
# and a chain of four is still a chain - lowering the floor from 5 to 4 recovers
# 12% more shafts in studyA without changing the median chain at all. The number
# is pinned here rather than left to the command line because a chain file whose
# parameters are not recorded cannot be reproduced: studyA_chains.geojson was
# once built with an unrecorded setting and the difference (275 vs 197 chains)
# took an hour to rediscover.
CHAIN_MIN_PTS = 4          # a chain needs at least this many shafts
CHAIN_MIN_SPAN_M = 120.0   # and must extend at least this far
CHAIN_EXTEND_MAX_M = 40.0  # axis is extended past the end shafts, capped here

# A chain is flagged "fragmented" when it spans a shaft-free stretch longer
# than both of these. Neither number is cosmetic: on the six hand-checked
# chains, the one the reviewer rejected had three holes of 105/182/136 m in a
# 1730 m chain, while every accepted chain stayed under 76 m. The ratio form
# matters because shaft spacing differs by site (16 m on studyA, 30 m on
# study2), so an absolute threshold alone would flag half of one site and none
# of the other.
CHAIN_GAP_FLAG_M = 80.0
CHAIN_GAP_RATIO = 4.0

# --- geometry -> engineering mapping ------------------------------------
# OSM highway tag -> (road class, formation width m)
# width follows JTG B01-2014 typical cross sections; edit here, not in code.
#
# These are **formation widths** (路基宽度): the whole cross-section the ground
# occupies, including hard shoulders, the median and the earth shoulders. They
# are NOT paved widths. A four-lane divided section with a 26 m formation
# carries about 23 m of pavement across both carriageways (4 x 3.75 lanes,
# 2 x 0.75 edge strips, 2 x 3.0 hard shoulders), and one carriageway of it is
# about 10.5 m. That is the same quantity an Iran transect returns when it
# measures 8.0-10.8 m of asphalt, so the two must not be compared directly:
# `qanat width` reports what changes when the value in this table is replaced by
# a paved width, and `paper/methods_width.md` states which quantity is used
# where.
ROAD_WIDTH = {
    "motorway":      ("expressway", 26.0),
    "trunk":         ("class1",     24.5),
    "primary":       ("class1",     24.5),
    "secondary":     ("class2",     12.0),
    "tertiary":      ("class3",      7.5),
    "unclassified":  ("class4",      6.5),
    "residential":   ("class4",      6.5),
    "service":       ("class4",      4.5),
    # below the county road: farm tracks, field paths, links. Adding these does
    # not change any existing class; it only gives a width to ways that used to
    # fall back to the 6.5 m default and would have been drawn too wide.
    "living_street": ("class4",      6.0),
    "road":          ("class4",      7.0),
    "track":         ("class4",      3.5),
    "cycleway":      ("class4",      2.5),
    "bridleway":     ("class4",      2.0),
    "path":          ("class4",      2.0),
    "footway":       ("class4",      1.5),
    "steps":         ("class4",      1.5),
    "corridor":      ("class4",      1.5),
    "construction":  ("class4",      5.0),
    "proposed":      ("class4",      5.0),
    "motorway_link": ("class3",     12.0),
    "trunk_link":    ("class3",     12.0),
    "primary_link":  ("class3",     10.0),
    "secondary_link": ("class3",    10.0),
    "tertiary_link": ("class4",      8.0),
}

# Statutory building-control distance from road edge, m.
# Regulation on the Protection of Highway Safety (State Council Order 593),
# Article 11: expressway >=30, national >=20, provincial >=15, county >=10,
# township >=5. Kept as the *baseline for comparison*, not as a target.
BASELINE_BUFFER_M = {
    "expressway": 30.0,
    "class1":     20.0,
    "class2":     15.0,
    "class3":     10.0,
    "class4":      5.0,
}

# Statutory distance by facility type. Same rule as above: these are quoted
# numbers, not fitted ones.
#   highway  -> Order 593 Art.11, by road grade (BASELINE_BUFFER_M)
#   railway  -> Regulation on Railway Safety Management (Order 639) Art.27,
#               measured from the toe of the embankment / crest of the cutting.
#               Zone-dependent: urban / suburban / village / other
#                  railway 8 / 10 / 12 / 15 m,  HSR 10 / 12 / 15 / 20 m.
#               The study area is open Gobi, so "other" is the default.
#   tunnel   -> deliberately undefined. An underground alignment has no surface
#               building-control line; what binds is the vertical clearance to
#               another buried work, which needs burial depth from both sides.
#               Do not invent a horizontal number for it.
RAILWAY_ZONE_M = {
    "urban":    {"railway": 8.0,  "hsr": 10.0},
    "suburban": {"railway": 10.0, "hsr": 12.0},
    "village":  {"railway": 12.0, "hsr": 15.0},
    "other":    {"railway": 15.0, "hsr": 20.0},
}


def buffer_for(facility: str, road_class: str | None = None, zone: str = "other"):
    """Statutory distance for a facility, in metres; None where none exists."""
    facility = str(facility).lower()
    if facility == "highway":
        return baseline_buffer(road_class or "expressway")
    if facility in ("railway", "hsr"):
        return RAILWAY_ZONE_M.get(zone, RAILWAY_ZONE_M["other"])[facility]
    return None


# --- hazard thresholds --------------------------------------------------
# DECLARED BUT UNUSED. These are the interface for the mechanics layer (T13:
# Boussinesq stress + Peck trough; T14: FEM cross-check), which is frozen until
# the burial-depth pathway is agreed - nothing in the toolkit can compute H yet,
# so a criterion built on H/b would be a number with no input. Kept here so the
# parameters are not quietly reinvented later; do not cite them as calibrated.
BURIAL_RATIO_COLLAPSE = 3.0    # documented collapse cases at H ~ 3 m
BURIAL_RATIO_STABLE = 8.5      # stable above this

# Status weight w: visual label -> mechanical reduction factor.
STATUS_WEIGHT = {
    "active":   1.00,   # still conveys water, cavity not yet expanded
    "inactive": 0.60,   # abandoned, long-term weathering / cavity growth
    "unknown":  0.80,
}

# Decay length for the distance kernel exp(-d / LAMBDA_D).
LAMBDA_D_M = 20.0

# --- runtime boundary ---------------------------------------------------
RUNTIME = {
    "detect":  "gpu",      # YOLO batch inference on AutoDL
    "samroad": "gpu",
    "prj":     "local",    # projection, chain rebuild, profile, routing
    "report":  "local",
}

# --- study areas --------------------------------------------------------
# name -> (lon_min, lat_min, lon_max, lat_max), WGS84 degrees
AOI = {
    # demo box around the Turpan oasis / G312 corridor, replace with real AOIs
    "turpan_demo": (89.05, 42.86, 89.30, 43.00),
    # placeholders to be filled from field records in P1
    "aoi_a": None,
    "aoi_b": None,
}


def road_class(highway: str):
    """Return (road_class, formation_width_m) for an OSM highway tag."""
    return ROAD_WIDTH.get(str(highway).lower(), ("class4", 6.5))


def baseline_buffer(road_cls: str) -> float:
    return BASELINE_BUFFER_M.get(road_cls, 5.0)
