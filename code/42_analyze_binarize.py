"""分析人工修图的灰度分布，为二值化定阈值（用户提醒：人工图可能有灰色噪点）。

目的:
  人工图是 Photoshop 手工倒背景的产物，可能残留灰色噪点（没倒干净的浅灰）。
  二值化阈值定低了 -> 噪点被当成根；定高了 -> 细根被抹掉。
  所以先看灰度直方图的形态，找出"根"与"灰噪"之间的谷。

产出: outputs/二值化分析/<样本>_hist.png + 阈值建议表
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# --- 统一路径（见 paths.py；数据搬家只改那里）---
sys.path.insert(0, str(Path(__file__).parent))
import paths as P  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
PROJ = P.PROJ
MAN = P.TEST_MAN
ORIG = P.TEST_ORIG
OUT = PROJ / "outputs" / "二值化分析"
SAMPLES = ["66-003", "28-CL-003", "G218-001", "126-003", "134-003", "7-003"]


def orig_of(stem: str) -> Path:
    for e in (".tif", ".jpg", ".png"):
        p = ORIG / (stem + e)
        if p.exists():
            return p
    return ORIG / (stem + ".tif")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"{'样本':<12}{'前景%':>9}{'中位灰度':>10}{'p10':>7}{'p50':>7}{'p90':>7}"
          f"{'灰噪(180-250)%':>15}{'建议阈值':>10}")
    for s in SAMPLES:
        p = MAN / f"{s}-man.tif"
        if not p.exists():
            print(f"  {s}: 缺文件")
            continue
        g = np.asarray(Image.open(p).convert("L"))
        nonwhite = g < 250
        vals = g[nonwhite]
        if vals.size == 0:
            continue
        # 灰噪 = 介于"明显是根"和"纯白"之间的浅灰
        gray_noise = ((g >= 180) & (g < 250)).mean() * 100
        # 阈值: 在 110~200 之间找"前景像素数随阈值变化最平缓"处不合适，
        # 改用分位数法 —— 取前景灰度分布的 p85 作为初值，再看直方图谷
        thr = int(np.percentile(vals, 85))
        rows.append({"样本": s, "前景占比%": round(100 * nonwhite.mean(), 3),
                     "中位灰度": round(float(np.median(vals)), 1),
                     "p10": round(float(np.percentile(vals, 10)), 1),
                     "p50": round(float(np.percentile(vals, 50)), 1),
                     "p90": round(float(np.percentile(vals, 90)), 1),
                     "灰噪(180-250)%": round(gray_noise, 3),
                     "建议阈值": thr})
        print(f"{s:<12}{100*nonwhite.mean():>9.3f}{np.median(vals):>10.1f}"
              f"{np.percentile(vals,10):>7.1f}{np.percentile(vals,50):>7.1f}"
              f"{np.percentile(vals,90):>7.1f}{gray_noise:>15.3f}{thr:>10}")

        # 直方图
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
        ax[0].hist(vals, bins=256, range=(0, 255), color="#444")
        ax[0].axvline(thr, color="r", ls="--", label=f"建议阈值 {thr}")
        ax[0].set_title(f"{s}-man  前景像素灰度分布")
        ax[0].set_yscale("log"); ax[0].legend()
        ax[1].hist(g.ravel(), bins=256, range=(0, 255), color="#888")
        ax[1].set_yscale("log"); ax[1].set_title("全图灰度分布（看灰噪平台）")
        for a in ax:
            a.set_xlabel("灰度")
        fig.tight_layout()
        fig.savefig(OUT / f"{s}_hist.png", dpi=110)
        plt.close(fig)

    with (OUT / "阈值建议.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"\n  -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
