"""阶段一 · 报告用汇总统计 + 测量换算速查表。

产出:
  outputs/阶段一/report_stats.json
  outputs/阶段一/measure_lookup.csv   逐张: 尺寸/dpi/毫米每像素（后续根长根面积换算直接用）
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

OUT = Path(r"D:\根系分割项目\outputs\阶段一")
EXCLUDE = {"树根004.jpg"}  # 透射背光异常件


def main() -> None:
    rows = list(csv.DictReader((OUT / "survey.csv").open(encoding="utf-8-sig")))
    normal = [r for r in rows if Path(r["rel_path"]).name not in EXCLUDE]

    def col(rs, k):
        return np.array([float(r[k]) for r in rs])

    stats = {
        "n_total": len(rows),
        "n_normal": len(normal),
        "outliers": [r["rel_path"] for r in rows if Path(r["rel_path"]).name in EXCLUDE],
        "battery": {},
    }
    for k in ["proxy_shadow_pct", "proxy_root_pct", "dark_pct", "border_dark_pct",
              "bg_grid_range", "s_mean", "v_bg", "v_p05"]:
        v = col(normal, k)
        stats["battery"][k] = {
            "p05": round(float(np.percentile(v, 5)), 4),
            "p50": round(float(np.percentile(v, 50)), 4),
            "p95": round(float(np.percentile(v, 95)), 4),
            "max": round(float(v.max()), 4),
            "argmax": Path(normal[int(v.argmax())]["rel_path"]).name,
        }

    bd = col(normal, "border_dark_pct")
    stats["border_dark_gt15"] = int((bd > 15).sum())
    stats["border_dark_gt25"] = int((bd > 25).sum())
    sh = col(normal, "proxy_shadow_pct")
    stats["shadow_gt10"] = int((sh > 10).sum())
    stats["shadow_gt15"] = int((sh > 15).sum())
    stats["dpi_distribution"] = {"600": sum(1 for r in normal if r["dpi_x"] == "600.0"),
                                 "720": sum(1 for r in normal if r["dpi_x"] == "720.0")}
    (OUT / "report_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False),
                                           encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    with (OUT / "measure_lookup.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.writer(fh)
        wr.writerow(["rel_path", "batch", "width_px", "height_px", "dpi", "mm_per_px",
                     "scan_width_mm", "scan_height_mm", "px_per_mm"])
        for r in rows:
            dpi = float(r["dpi_x"] or 600)
            mmpp = 25.4 / dpi
            wr.writerow([r["rel_path"], r["batch"], r["width"], r["height"], r["dpi_x"],
                         round(mmpp, 6), round(int(r["width"]) * mmpp, 2),
                         round(int(r["height"]) * mmpp, 2), round(dpi / 25.4, 4)])
    print(f"\n[measure_lookup.csv] 已写出 {len(rows)} 行")


if __name__ == "__main__":
    main()
