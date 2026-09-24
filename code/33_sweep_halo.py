"""用干净真值（7-001/2/3）扫参数：贴根半影外扩量到底该设多少。

背景:
  之前按"贴根半影计入根系、可适当修剪"设了 HALO_GROW=6px。
  但用用户刚修完的真值一测: 我的根面积是用户的 1.6~1.9 倍，精确率只有 0.5~0.63
  —— 说明 6px 外扩把根显著加粗了，会直接导致根径高估 40~90%。

本脚本扫 HALO_GROW ∈ {0,1,2,3,4,6} 与 GRAY_HALO，用真值算 IoU/召回/精确，
选出"不牺牲召回的前提下精确率最高"的取值。
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import rootseg_common as R  # noqa: E402

# --- 统一路径（见 paths.py；数据搬家只改那里）---
sys.path.insert(0, str(Path(__file__).parent))
import paths as P  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
PROJ = P.PROJ
GT_DIR = P.TEST_MAN_99GAI
ORIG_DIR = P.RAW_D2_99
OUT = PROJ / "outputs" / "手工真值评估"
SAMPLES = ["7-001", "7-002", "7-003"]      # 用户明确说这三张是精修过的
HALOS = [0, 1, 2, 3, 4, 6]
NONWHITE = 250


def main() -> int:
    gts, origs = {}, {}
    for s in SAMPLES:
        g = GT_DIR / (s + ".tif")
        if not g.exists():
            g = next(GT_DIR.glob(s + ".*"))
        o = ORIG_DIR / (s + ".tif")
        if not o.exists():
            o = next(ORIG_DIR.glob(s + ".*"))
        gts[s] = np.any(np.asarray(Image.open(g).convert("RGB")) < NONWHITE, axis=2)
        origs[s] = np.asarray(Image.open(o).convert("RGB"))
        print(f"  载入 {s}: 真值根面积 {100*gts[s].mean():.3f}%")

    rows = []
    print(f"\n{'HALO':>5}{'IoU':>9}{'召回':>9}{'精确':>9}{'漏检%':>9}{'我的面积/真值':>13}")
    for halo in HALOS:
        R.HALO_GROW = halo
        ious, recs, pres, mys, gtsz = [], [], [], [], []
        for s in SAMPLES:
            root, _, _ = R.extract_root(origs[s], return_maps=False)
            gt = gts[s]
            inter = int((gt & root).sum())
            ious.append(inter / max(1, int((gt | root).sum())))
            recs.append(inter / max(1, int(gt.sum())))
            pres.append(inter / max(1, int(root.sum())))
            mys.append(float(root.mean()))
            gtsz.append(float(gt.mean()))
        n = len(SAMPLES)
        row = {"HALO_GROW": halo,
               "IoU": round(sum(ious) / n, 4), "recall": round(sum(recs) / n, 4),
               "precision": round(sum(pres) / n, 4),
               "漏检%": round(100 * (1 - sum(recs) / n), 2),
               "面积倍数": round((sum(mys) / n) / (sum(gtsz) / n), 3)}
        rows.append(row)
        print(f"{halo:>5}{row['IoU']:>9.4f}{row['recall']:>9.4f}{row['precision']:>9.4f}"
              f"{row['漏检%']:>9.2f}{row['面积倍数']:>13.3f}")

    depth_csv = OUT / "halo_sweep.csv"
    with depth_csv.open("w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    good = [r for r in rows if r["recall"] >= 0.97]
    best = max(good or rows, key=lambda r: r["precision"])
    print(f"\n  召回 >= 0.97 的取值里，精确率最高的是 HALO_GROW = {best['HALO_GROW']}px")
    print(f"    IoU {best['IoU']:.4f}  召回 {best['recall']:.4f}  精确 {best['precision']:.4f}  "
          f"面积倍数 {best['面积倍数']:.3f}")
    print(f"  表 -> {depth_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
