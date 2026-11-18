"""qanat - shaft-chain / alignment coupling toolkit.

Contract (do not change without bumping SCHEMA_VERSION):
  * exchange format : WGS84 GeoJSON FeatureCollection, 3 geometries only
  * working CRS     : EPSG:32645 (UTM 45N), projected exactly once at entry
  * runtime boundary: local | gpu | manual, cross-boundary = files only
"""

__version__ = "0.1.0"
SCHEMA_VERSION = "1.1"
