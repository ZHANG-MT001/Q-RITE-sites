# Data licence and attribution

This repository holds three kinds of material, and they are not under the same
terms. Read this before redistributing any of it.

## 1. Code — MIT

`index.html`, `router.js`, `algorithms/`, `vendor/` and the build notes in
`TECHNICAL.md` are released under the MIT licence; see `LICENSE`.

## 2. Derived vector layers — CC BY 4.0

The shaft points, the traced rows and the crossing table in `sites/` are
derived products of the Q-RITE study. They are released under Creative Commons
Attribution 4.0 International (CC BY 4.0). Cite as:

> Zhang M., Zhang Y. Q-RITE: Qanat Shaft-Row Viewer (v1.0.1) [Data and software].
> 2026. Zenodo. https://doi.org/10.5281/zenodo.24432398

Two upstream obligations travel with these layers and must be kept in any
redistribution:

- **Roads** come from OpenStreetMap and remain © OpenStreetMap contributors,
  under the Open Database Licence (ODbL 1.0). Any derivative of the road
  geometry must stay ODbL.
- **Shaft/row vectors** were produced by a detector fine-tuned on the
  *Qanat-ActivityStates-YOLO* dataset (Fu X. et al., 2026, Zenodo,
  doi:10.5281/zenodo.18636032, CC BY 4.0). Attribution to that deposit is
  required when the detector rather than the vectors is discussed.

## 3. Imagery — not licensed for redistribution

The background tiles are Bing Maps imagery, © Microsoft. They are included in
the demonstration page only so that a reader can see what the detector saw. They
are **not** covered by the MIT or CC BY grants above, and Microsoft's terms do
not permit redistribution of the tiles as a dataset.

Two consequences, and the second is the one to act on:

1. Anyone reusing this repository for anything other than reading the page
   should not extract the tiles.
2. If the page is to be cited as a citable research object — which it now is —
   the safest course is to ship it **without** the imagery, or with a
   replaceable basemap (OSM raster tiles, or a plain graticule), and to keep the
   Bing mosaic as an optional local overlay the user adds themselves.
   Otherwise the repository redistributes third-party imagery that it has no
   right to redistribute, and a reviewer or an editor may say so.

A one-line switch is usually enough: draw the vectors on a blank canvas, and
load an overlay only when a local file is present.

## Provenance

| Layer | Source | Terms |
|---|---|---|
| Shaft points, rows, crossings | Q-RITE detector and tracer | CC BY 4.0 (this repository) |
| Roads | OpenStreetMap, 2026-11 snapshot via Overpass | ODbL 1.0 |
| Background tiles | Bing Maps | © Microsoft, demonstration only |
| Training data | Qanat-ActivityStates-YOLO (Fu et al., 2026) | CC BY 4.0 |
