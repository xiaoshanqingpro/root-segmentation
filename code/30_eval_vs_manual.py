"""用用户的手工标注评估当前管线 —— 终于有真值了。

数据:
  真值 = 采样数据\图片\多样性林\9.9改\<样本>   （用户手工倒背景：根保留、背景倒白）
  待评 = 我的管线跑在 9.9 原图上产出的白底根图

两者同为 7019x4962 @600dpi，像素对齐，可直接比。

指标（全部是真实误差，不是代理量）:
  IoU / Dice        根掩膜的重合度
  召回 recall       用户标为根、我也判为根的比例  -> 低 = 我漏了真根
  精确 precision    我判为根、用户也标为根的比例  -> 低 = 我把非根当成了根
  漏检像素 / 误检像素 及各自占比

产出: outputs/手工真值评估/
        metrics.csv                   逐张指标
        <样本>_eval.png               原图 | 用户真值 | 我的结果 | 误差图(红=误检 蓝=漏检)
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

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

NONWHITE_GT = 250       # 用户真值里 < 此值算"根"
NONWHITE_ME = 250       # 我的结果里 < 此值算"根"


def font(sz):
    for n in ("msyh.ttc", "simhei.ttf"):
        try:
            return ImageFont.truetype(n, sz)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


def nonwhite_mask(arr: np.ndarray, thr: int) -> np.ndarray:
    return np.any(arr < thr, axis=2)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    gts = sorted(GT_DIR.glob("*"))
    print(f"[手工真值评估] 真值 {len(gts)} 张，来自 {GT_DIR}")
    rows = []
    for g in gts:
        o = ORIG_DIR / (g.stem + ".tif")
        if not o.exists():
            cand = list(ORIG_DIR.glob(g.stem + ".*"))
            if not cand:
                print(f"  !! 找不到 {g.stem} 的原图，跳过")
                continue
            o = cand[0]

        gt_arr = np.asarray(Image.open(g).convert("RGB"))
        me_arr = np.asarray(Image.open(o).convert("RGB"))
        print(f"  处理 {g.name}  (原图 {o.name})", flush=True)
        my_root, _, info = R.extract_root(me_arr, return_maps=False)

        gt = nonwhite_mask(gt_arr, NONWHITE_GT)
        if gt.shape != my_root.shape:
            print(f"    尺寸不一致 {gt.shape} vs {my_root.shape}，跳过")
            continue

        inter = int((gt & my_root).sum())
        union = int((gt | my_root).sum())
        iou = inter / max(1, union)
        dice = 2 * inter / max(1, int(gt.sum()) + int(my_root.sum()))
        recall = inter / max(1, int(gt.sum()))
        prec = inter / max(1, int(my_root.sum()))
        fn_mask = gt & ~my_root          # 我漏掉的真根
        fp_mask = ~gt & my_root          # 我多判的非根
        fn = int(fn_mask.sum())
        fp = int(fp_mask.sum())
        print(f"    真值根面积 {100*gt.mean():.3f}%   我的根面积 {100*my_root.mean():.3f}%"
              f"   (倍数 {my_root.mean()/max(1e-9, gt.mean()):.2f}x)")
        rows.append({"样本": g.stem, "真值来源": g.name, "原图": o.name,
                     "gt_root_pct": round(100 * gt.mean(), 4),
                     "my_root_pct": round(100 * my_root.mean(), 4),
                     "IoU": round(iou, 4), "Dice": round(dice, 4),
                     "recall": round(recall, 4), "precision": round(prec, 4),
                     "漏检px": fn, "漏检占真值%": round(100 * fn / max(1, int(gt.sum())), 2),
                     "误检px": fp, "误检占我的结果%": round(100 * fp / max(1, int(my_root.sum())), 2)})
        print(f"    IoU {iou:.3f}  Dice {dice:.3f}  召回 {recall:.3f}  精确 {prec:.3f}  "
              f"漏检 {100*fn/max(1,int(gt.sum())):.1f}%  误检 {100*fp/max(1,int(my_root.sum())):.1f}%")

        # 误差图
        H, W = gt.shape
        rgb = me_arr.copy()
        vis = rgb.astype(np.float32)
        vis[gt & my_root] = vis[gt & my_root] * 0.4 + np.array([0, 180, 0]) * 0.6
        vis[fp_mask] = vis[fp_mask] * 0.25 + np.array([255, 30, 30]) * 0.75      # 误检=红
        vis[fn_mask] = vis[fn_mask] * 0.25 + np.array([40, 90, 255]) * 0.75      # 漏检=蓝
        panels = [me_arr, np.stack([gt * 255] * 3, -1).astype(np.uint8),
                  np.stack([my_root * 255] * 3, -1).astype(np.uint8), vis.astype(np.uint8)]
        PW = 620
        tiles = [cv2.resize(p, (PW, int(p.shape[0] * PW / p.shape[1])), interpolation=cv2.INTER_AREA)
                 for p in panels]
        th = max(t.shape[0] for t in tiles)
        row = np.hstack([np.pad(t, ((0, th - t.shape[0]), (0, 0), (0, 0))) for t in tiles])
        c = Image.new("RGB", (row.shape[1], row.shape[0] + 46), (255, 255, 255))
        c.paste(Image.fromarray(row), (0, 46))
        dr = ImageDraw.Draw(c)
        dr.text((8, 6), f"手工真值评估  {g.stem}   IoU {iou:.3f}  Dice {dice:.3f}  "
                        f"召回 {recall:.3f}  精确 {prec:.3f}", fill=(0, 0, 0), font=font(22))
        for i, t in enumerate(["原图", "用户手工真值", "我的结果", "误差(红=误检 蓝=漏检)"]):
            dr.text((8 + i * PW, 26), t, fill=(150, 0, 0), font=font(17))
        c.save(OUT / f"{g.stem}_eval.png", optimize=True)

    if rows:
        with (OUT / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
        import statistics as st
        print("\n=== 汇总（中位）===")
        for k in ["IoU", "Dice", "recall", "precision"]:
            print(f"  {k:<10} {st.median(r[k] for r in rows):.4f}")
        print(f"  漏检占真值   中位 {st.median(r['漏检占真值%'] for r in rows):.2f}%")
        print(f"  误检占我的结果 中位 {st.median(r['误检占我的结果%'] for r in rows):.2f}%")
        print(f"\n  指标表 -> {OUT/'metrics.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
