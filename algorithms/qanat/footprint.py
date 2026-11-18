"""How much footprint overlap the centreline count leaves out.

Section 5.2 reports a gap that the crossing table cannot show. The sweep counts
contacts at the road centreline, but the pavement is a band, so a row that runs
inside the band without crossing the centreline is invisible to that count. The
paired tables in ``data/out/footprint_{site}.csv`` hold both definitions for all
540 unique alignments of each area: what the centreline-crossing events
accumulate, and what intersecting every row axis with the full 26 m by 3 km
rectangle accumulates.

The gap is not small and it is not uniform:

    area A   277.7 km by crossings against 304.1 km by footprint, 8.7% omitted
    area B   104.2 km against 112.5 km, 7.4% omitted

The zero-crossing end is where it bites hardest, because that is the end an
operator would read as "no exposure": area A alignment b117.0_o-00200 crosses no
centreline and still carries 70.4 m of footprint overlap from two rows.

    python -m qanat footprint
"""

from __future__ import annotations

import csv
from pathlib import Path

SRC = "data/out/footprint_{site}.csv"
SITES = (("studyA", "A"), ("study2", "B"))
OUT_CSV = Path("paper/t12_footprint_gap.csv")
OUT_MD = Path("paper/results_footprint_gap.md")
COLS = ("plan", "bearing_deg", "offset_m", "centerline_crossings",
        "crossing_only_length_m", "all_footprint_overlap_length_m",
        "omitted_overlap_length_m", "all_footprint_spacing_equivalents",
        "overlapping_chains")


def _rows(site: str) -> list[dict]:
    with open(SRC.format(site=site), encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def one(site: str, tag: str) -> dict:
    rows = _rows(site)
    miss = [r for r in rows if r.get("plan") is None]      # none expected
    assert not miss, f"{site}: {len(miss)} short rows"
    co = sum(float(r["crossing_only_length_m"]) for r in rows)
    fp = sum(float(r["all_footprint_overlap_length_m"]) for r in rows)
    om = sum(float(r["omitted_overlap_length_m"]) for r in rows)
    zero = [r for r in rows if int(r["centerline_crossings"]) == 0]
    zero_hit = [r for r in zero if float(r["all_footprint_overlap_length_m"]) > 1.0]
    pos = [r for r in rows if float(r["all_footprint_overlap_length_m"]) > 0]
    frac = sorted(float(r["omitted_overlap_length_m"])
                  / float(r["all_footprint_overlap_length_m"]) for r in pos)
    n = len(frac)
    return {
        "site": tag, "plans": len(rows),
        "crossing_km": co / 1000.0, "footprint_km": fp / 1000.0,
        "omitted_km": om / 1000.0,
        "omitted_share": om / fp if fp else 0.0,
        "zero_plans": len(zero),
        "zero_with_overlap": len(zero_hit),
        "zero_max_m": max((float(r["all_footprint_overlap_length_m"])
                           for r in zero), default=0.0),
        "per_plan_med": frac[n // 2] if frac else 0.0,
        "per_plan_p90": frac[int(0.9 * (n - 1))] if frac else 0.0,
        "worst_plan": max(rows, key=lambda r: float(r["omitted_overlap_length_m"]))["plan"],
        "worst_m": max(float(r["omitted_overlap_length_m"]) for r in rows),
    }


def run() -> list[dict]:
    return [one(s, t) for s, t in SITES]


def write(rows: list[dict]) -> list[Path]:
    OUT_CSV.parent.mkdir(exist_ok=True)
    keys = list(rows[0].keys())
    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(keys)
        for r in rows:
            w.writerow([f"{r[k]:.4f}" if isinstance(r[k], float) else r[k] for k in keys])
    a, b = rows
    md = "\n".join([
        "# 中心线交叉累计 vs 完整足迹重叠",
        "",
        f"数据源：`data/out/footprint_studyA.csv`、`footprint_study2.csv`"
        f"（每区 540 条唯一线位，3 km × 26 m 矩形足迹）。",
        "",
        "| | A 区 | B 区 |",
        "|---|---|---|",
        f"| 中心线交叉累计的重叠总长 | {a['crossing_km']:.1f} km | {b['crossing_km']:.1f} km |",
        f"| 完整足迹的重叠总长 | {a['footprint_km']:.1f} km | {b['footprint_km']:.1f} km |",
        f"| 只计交叉漏掉的长度 | {a['omitted_km']:.2f} km | {b['omitted_km']:.2f} km |",
        f"| 漏计占比（求和之比） | {a['omitted_share'] * 100:.1f}% | "
        f"{b['omitted_share'] * 100:.1f}% |",
        f"| 单条线位漏计比例中位 / P90 | {a['per_plan_med'] * 100:.1f}% / "
        f"{a['per_plan_p90'] * 100:.1f}% | {b['per_plan_med'] * 100:.1f}% / "
        f"{b['per_plan_p90'] * 100:.1f}% |",
        f"| 零交叉线位 | {a['zero_plans']} | {b['zero_plans']} |",
        f"| 其中足迹内仍有 >1 m 重叠 | {a['zero_with_overlap']} | {b['zero_with_overlap']} |",
        f"| 零交叉线位的最大足迹重叠 | {a['zero_max_m']:.1f} m | {b['zero_max_m']:.1f} m |",
        f"| 漏计最多的线位 | {a['worst_plan']}（{a['worst_m']:.0f} m） | "
        f"{b['worst_plan']}（{b['worst_m']:.0f} m） |",
        "",
        "## 结论口径",
        "",
        f"1. 只累加中心线交叉事件，会漏掉该区完整足迹重叠的 "
        f"{a['omitted_share'] * 100:.1f}%（A）与 {b['omitted_share'] * 100:.1f}%（B）。"
        "这个比例是跨线位求和之比，不是单条路线的风险误差。",
        f"2. 漏计不是均匀摊派：单条线位上从中位 {a['per_plan_med'] * 100:.1f}% "
        f"到 P90 {a['per_plan_p90'] * 100:.1f}%，即有些线位几乎全漏。",
        f"3. 零交叉不等于零暴露：A 区 {a['zero_plans']} 条零交叉线位里 "
        f"{a['zero_with_overlap']} 条在足迹内仍有重叠，最大 {a['zero_max_m']:.1f} m。",
        "",
    ])
    OUT_MD.write_text(md, encoding="utf-8")
    return [OUT_CSV, OUT_MD]


def main(log=print):
    rows = run()
    for p in write(rows):
        log(f"wrote {p}")
    for r in rows:
        log(f"  {r['site']}: {r['omitted_share'] * 100:.1f}% omitted "
            f"({r['omitted_km']:.2f} of {r['footprint_km']:.1f} km), "
            f"{r['zero_with_overlap']}/{r['zero_plans']} zero-crossing plans still overlap")
    return rows


if __name__ == "__main__":
    main()
