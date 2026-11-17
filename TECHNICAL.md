# Technical notes · 技术说明

A static site: one HTML page, one small routing script, vendored Leaflet, and all data inline. Nothing is fetched, so it runs from `file://` or any static host (including GitHub Pages).

这是一个纯静态站点：一个 HTML 页面、一个选线脚本、本地 Leaflet、全部数据内联。运行时不发起任何网络请求，因此 `file://` 直接打开或任意静态托管（含 GitHub Pages）都能用。

## Files · 文件

| File | Role · 作用 |
|---|---|
| `index.html` | UI and all page logic · 界面与页面逻辑 |
| `router.js` | Route engine: A* + smoothing + scoring · 选线引擎：A*、平滑、评分 |
| `sites/data.js` | `window.QSITES` — inline JSON, the only data source · 内联 JSON，唯一数据源 |
| `sites/*.jpg` | Image overlays, one per window · 每窗口一张影像底图 |
| `vendor/leaflet.*` | Leaflet, copied locally (no CDN) · 本地 Leaflet，无 CDN |

## Why data is a `<script>` · 为什么数据放进 `<script>`

`fetch()` is blocked under `file://`. The five windows are therefore packed as `window.QSITES` in `sites/data.js` and loaded with a `<script>` tag — the same bytes work online and offline.

`file://` 下 `fetch()` 被浏览器拦截，所以五个窗口打成 `sites/data.js` 里的 `window.QSITES`，用 `<script>` 加载——同一份数据在线离线通用。

## Route tool · 选线

The expensive part (shaft distance field + two hard masks) is precomputed and base64-packed into `data.js`. The browser only solves the two endpoints the user clicks, with A*, Chaikin smoothing and a weighted score `(L/L₀−1) + C/C₀`. Results differ from the server build by ~0.5–1% because of grid origin.

井链距离场与两个硬掩膜预先算好、base64 打包进 `data.js`；浏览器只对用户点的两个端点跑 A*、Chaikin 平滑与加权评分 `(L/L₀−1) + C/C₀`。因网格起点不同，与服务器版差约 0.5–1%。

## Data & credits · 数据与署名

- Shafts / rows: Q-RITE detection + tracing output · 井口/井链：Q-RITE 检测与追踪结果
  - Detections come from a YOLOv8s fine-tuned on the public *Qanat-ActivityStates-YOLO* dataset (Fu et al. 2026, [doi:10.5281/zenodo.18636032](https://doi.org/10.5281/zenodo.18636032), CC BY 4.0); the five windows were delimited after training · 井口由 YOLOv8s 在公开数据集 Qanat-ActivityStates-YOLO 上微调后检出，五窗为训练后圈定
- Imagery: Bing Maps tiles (© Microsoft) · 影像：Bing Maps 瓦片
- Roads: OpenStreetMap (© OSM contributors, ODbL) · 道路：OpenStreetMap

Only derived vectors and demo image overlays ship here; the source imagery is not redistributed. · 此处只带派生矢量与演示底图，不分发原始影像。

## Rebuild · 重新生成

From the full research repository (not needed to view): `python -m qanat static` recomputes `sites/data.js` and the five image overlays from the project GeoJSON/CSV. The page itself is a hand-written template and is never overwritten.

在完整研究仓库中（浏览不需要）：`python -m qanat static` 从项目 GeoJSON/CSV 重算 `sites/data.js` 与五张底图；页面是手写模板，不会被覆盖。
