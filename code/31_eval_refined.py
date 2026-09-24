"""手工真值的成色分析 + 剥离污染后重算指标。

上一版评估得到 IoU 0.23 / 召回 0.25，但误差图暴露: 蓝色(漏检)里很大一块是
**图像四周一整圈边框**，以及背景里未清完的**水渍灰斑**。
用户也说过那批是"修了一半的图片"。所以直接拿它当真值会**低估**我的管线。

本脚本做两件事:
  1. 拆解真值的成分: 边框框、浅灰污染、深色真根，各占多少
  2. 剔除污染后重算指标: 裁掉外圈 N 像素；并且只把"深色"的 GT 像素当真根
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

TRIM = 150          # 评估时裁掉的外圈像素（把未清理的扫描边框排除）
DARK_GT = 170       # 真值里灰度 < 此值才算"深色真根"；比这亮的是未清完的浅灰污染
NONWHITE = 250


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"[真值成色分析] 外圈裁 {TRIM}px，深色真根阈值 灰度<{DARK_GT}")
    for g in sorted(GT_DIR.glob("*")):
        o = ORIG_DIR / (g.stem + ".tif")
        if not o.exists():
            c = list(ORIG_DIR.glob(g.stem + ".*"))
            if not c:
                continue
            o = c[0]
        gt_rgb = np.asarray(Image.open(g).convert("RGB"))
        gt_gray = np.asarray(Image.open(g).convert("L")).astype(np.float32)
        me_rgb = np.asarray(Image.open(o).convert("RGB"))

        gt_all = np.any(gt_rgb < NONWHITE, axis=2)
        H, W = gt_all.shape

        # 成色拆解
        border = np.zeros_like(gt_all)
        border[:TRIM, :] = border[-TRIM:, :] = border[:, :TRIM] = border[:, -TRIM:] = True
        gt_border = gt_all & border
        gt_inner = gt_all & ~border
        gt_dark = gt_inner & (gt_gray < DARK_GT)
        gt_light = gt_inner & (gt_gray >= DARK_GT)

        my_root, _, info = R.extract_root(me_rgb, return_maps=False)
        my_inner = my_root & ~border

        def m(a, b, tag):
            inter = int((a & b).sum())
            iou = inter / max(1, int((a | b).sum()))
            rec = inter / max(1, int(a.sum()))
            pre = inter / max(1, int(b.sum()))
            return {"口径": tag, "GT": int(a.sum()), "ME": int(b.sum()),
                    "IoU": round(iou, 4), "recall": round(rec, 4), "precision": round(pre, 4),
                    "漏检%": round(100 * (a & ~b).sum() / max(1, int(a.sum())), 2),
                    "误检%": round(100 * (~a & b).sum() / max(1, int(b.sum())), 2)}

        r_raw = m(gt_all, my_root, "原口径（含边框+浅灰污染）")
        r_trim = m(gt_inner, my_inner, "裁边框后")
        r_dark = m(gt_dark, my_inner, "裁边框+只取深色真根")
        for r in (r_raw, r_trim, r_dark):
            r["样本"] = g.stem
        rows += [r_raw, r_trim, r_dark]

        n_all = max(1, int(gt_all.sum()))
        print(f"\n  {g.stem}   真值总像素 {n_all}")
        print(f"    边框污染 {100*gt_border.sum()/n_all:5.1f}%   "
              f"浅灰污染 {100*gt_light.sum()/n_all:5.1f}%   深色真根 {100*gt_dark.sum()/n_all:5.1f}%")
        print(f"    原口径   IoU {r_raw['IoU']:.3f}  召回 {r_raw['recall']:.3f}  精确 {r_raw['precision']:.3f}")
        print(f"    裁边框后 IoU {r_trim['IoU']:.3f}  召回 {r_trim['recall']:.3f}  精确 {r_trim['precision']:.3f}")
        print(f"    只算深色 IoU {r_dark['IoU']:.3f}  召回 {r_dark['recall']:.3f}  精确 {r_dark['precision']:.3f}")

    with (OUT / "metrics_refined.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    import statistics as st
    print("\n=== 三种口径的中位数对比 ===")
    print(f"{'口径':<26}{'IoU':>8}{'召回':>8}{'精确':>8}{'漏检%':>8}")
    for tag in ["原口径（含边框+浅灰污染）", "裁边框后", "裁边框+只取深色真根"]:
        sub = [r for r in rows if r["口径"] == tag]
        print(f"{tag:<26}{st.median(r['IoU'] for r in sub):>8.3f}"
              f"{st.median(r['recall'] for r in sub):>8.3f}"
              f"{st.median(r['precision'] for r in sub):>8.3f}"
              f"{st.median(r['漏检%'] for r in sub):>8.1f}")
    print(f"\n  表 -> {OUT/'metrics_refined.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
