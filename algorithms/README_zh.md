# Q-RITE 算法源码

[English README](README.md) · [Q-RITE 地图](../index.html) · [平台技术说明](../TECHNICAL.md)

本目录汇总 Q-RITE 的井链重建、公路几何暴露计算和候选线路比选代码。五窗口地图加载预先计算好的井链和既有道路交叉。用户点选起终点后，浏览器实时生成和评估候选线路。

## 算法索引

| 算法 | 代码入口 | 计算内容 |
| --- | --- | --- |
| 1. 道路与井链的交叉几何 | [`platform_crossings.assess`](qanat/platform_crossings.py)、[`angles.road_events`](qanat/angles.py) | 将道路中心线与推定暗渠轴线求交，输出交点、锐角夹角、重叠长度和按井距估算的等效井距数。 |
| 2. 避让代价场与候选线生成 | [`static_grid.build_site`](qanat/static_grid.py)、[`planRouteLocal`](browser/router.js)、[`route.build_density`](qanat/route.py) | 把井链栅格化，为网页预计算距离代价场和两档禁入掩膜。浏览器使用 A*，按不同避让权重生成候选线。研究模块另有一套 Python 代价场与搜索实现。 |
| 3. 绕行预算下的路线推荐 | [`pickBest`](browser/detour.js) | 以直线为长度基准，按绕行上限筛选候选方案。优先推荐估算重叠最短的方案，重叠相同时取较短路线。 |
| 4. 竖井点自动成链 | [`chains.rebuild`](qanat/chains.py)、[`rows.extract`](qanat/rows.py) | 用多尺度点对筛选和共线合并重建井链，或根据井排局部走向逐点追踪。 |
| 5. 沿井链补检与可靠性分级 | [`iran_refine.collect` 与 `merge`](qanat/iran_refine.py)、[`chains.grade_chain`](qanat/chains.py) | 沿井链及其端部外延搜索漏检点，按横向偏移和局部井距筛选。按拟合误差、井口数、覆盖率和长缺口给井链分级。 |
| 6. 完整道路足迹重叠 | [`cases.evaluate_plan`](qanat/cases.py)、[`footprint`](qanat/footprint.py) | 将推定井链轴线裁切到有限长宽的道路矩形内。论文比较了完整足迹重叠与仅计中心线交叉的重叠结果。 |
| 7. 沿线暴露剖面与接触分型 | [`profile.bins`](qanat/profile.py)、[`typology.contacts`](qanat/typology.py) | 按里程分段统计暴露，并区分正穿、斜穿、并行贴线、并行近接。 |
| 8. 影像横断测路宽 | [`wtransect.measure`](qanat/wtransect.py)、[`iran_crossing`](qanat/iran_crossing.py) | 横过已知道路采样亮度，以亮度凹槽的半深度估计路面边缘，并检查对比度、中心偏移与宽度稳定性。 |

## 运行示例

在本目录中使用 Python 3.10 或更高版本运行：

```bash
python -m pip install -r requirements.txt
python -B demo.py
```

示例使用合成坐标和合成亮度剖面。它重建一条井链、补回一个漏检井点，算出正交处 **26 m** 的重叠，以及一条中心线零交叉的平行线在道路足迹内仍有 **100 m** 的重叠。示例还会生成两档避让掩膜，并量得 **8 m** 的路宽。安装依赖后可离线运行。

## 调用方式

- **既有道路交叉。** 吐鲁番窗口可调用 `platform_crossings.assess(rows, ways, zone=45)`。井链提供投影后的端点 `p0`、`p1` 和以米计的中位井距 `spacing`。道路提供由 `[经度, 纬度]` 组成的 `pts`、以米计的宽度 `w`、道路类别 `cls` 和名称 `name`。
- **浏览器选线。** `browser/router.js` 是交付版的实时选线引擎。`window.planRouteLocal(waypoints, site)` 接收至少两个 `[经度, 纬度]` 点，以及与地图 `sites/data.js` 格式一致的窗口数据。窗口数据包含井链、`grid`、`field` 和打包的 `block` 掩膜。`browser/detour.js` 提供 `pickBest(candidates, budget)`，`0.10` 表示允许绕行 10%，`1` 对应地图里的不限绕行选项。候选列表需保留选线引擎生成的直线参照。
- **研究几何计算。** `chains.rebuild`、`rows.extract`、`cases.evaluate_plan`、`profile.bins` 和 `typology.contacts` 使用投影后的米制 NumPy 坐标。直线道路矩形的 `hw` 为半宽，26 m 路基对应 `hw=13.0`。
- **研究数据。** 原批处理函数沿用研究项目的 `data/` 目录结构。重跑论文表格或对真实影像测宽，需要对应的研究数据和影像缓存。二次检测还需要安装 Ultralytics，并由调用方提供适配的 YOLO 权重。已有亮度剖面时，可把取样函数作为 `profile_fn` 传给 `wtransect.measure`。

## 指标口径

网页按中心线交叉评分，用道路宽度除以夹角正弦估算重叠长度，并对极小角度作限制。拟建道路示例采用 26 m 路基宽。井链沿道路并行或靠近端部时，可用完整足迹计算检查实际落入道路矩形的映射长度。像 1.04 这样的值表示等效井距数。几何指标来自影像推定的井链轴线，用于初步选线和勘察筛查。

地图道路中心线来自 OpenStreetMap。竖井检测使用 YOLOv8s。A*、Liang–Barsky 裁切和 Chaikin 平滑属于实现中采用的通用方法。本目录收录源码和合成示例，真实影像和训练权重由对应研究流程另行提供。

## 源码说明

核心 Python 模块来自 Q-RITE 研究工具包。`platform_crossings.py` 和 `static_grid.py` 分别提取自 `live.py` 和 `static_build.py`。`browser/router.js` 来自交付版地图，`browser/detour.js` 将原页面的推荐逻辑提取为可显式传入绕行预算的函数。补检模块使用相对的默认权重路径，测宽模块支持向示例传入自定义亮度剖面。
