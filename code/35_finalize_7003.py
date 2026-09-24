"""7-003 定稿：让自动结果贴合用户精修真值的口径，并清掉水泡细线。

用户已确认:
  - "以你精修的真值为准，我继续收紧"
  - "优先把003的修出，我给老师过目"

步骤:
  1. 扫"收边"参数（HALO_GROW=0 + SHRINK_PX=0..4），选 IoU 最高的取值
  2. 用该取值跑 7-003 原图 -> 根掩膜 -> 白底根图
  3. 在白底图上做细壁闭合环清理（用户反馈的水泡细线）
  4. 输出成品（高质 JPEG + PNG）+ 与真值的对比图
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

import sys
sys.path.insert(0, str(Path(__file__).parent))
import rootseg_common as R  # noqa: E402

# --- 统一路径（见 paths.py；数据搬家只改那里）---
sys.path.insert(0, str(Path(__file__).parent))
import paths as P  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
PROJ = P.PROJ
ORIG = P.RAW_D2_99 / "7-003.tif"
GT = P.TEST_MAN_99GAI / "7-003.tif"
OUT = PROJ / "outputs" / "7-003三方对比"
NONWHITE = 250
MAX_WALL_ERODE = 3
MIN_HOLE_PX = 300


def font(sz):
    for n in ("msyh.ttc", "simhei.ttf"):
        try:
            return ImageFont.truetype(n, sz)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


def remove_loops(rgb):
    """细壁闭合环清理（用户反馈的水泡细线）"""
    fg = np.any(rgb < NONWHITE, axis=2)
    H, W = fg.shape
    ker = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    filled = ndimage.binary_fill_holes(fg)
    holes = filled & ~fg
    drop = np.zeros_like(fg)
    if not holes.any():
        return drop, 0
    nh, lh, sh, _ = cv2.connectedComponentsWithStats(holes.astype(np.uint8), 8)
    big = [(j, sh[j, cv2.CC_STAT_AREA]) for j in range(1, nh) if sh[j, cv2.CC_STAT_AREA] >= MIN_HOLE_PX]
    cnt = 0
    for r in range(1, MAX_WALL_ERODE + 1):
        if not big:
            break
        er = cv2.erode(fg.astype(np.uint8), ker(r)) > 0
        holes_er = ndimage.binary_fill_holes(er) & ~er
        still = []
        for j, area in big:
            m = lh == j
            ys, xs = np.nonzero(m)
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            if holes_er[y0:y1, x0:x1].any():
                still.append((j, area))
            else:
                pad = r + 2
                yy0, yy1 = max(0, y0 - pad), min(H, y1 + pad)
                xx0, xx1 = max(0, x0 - pad), min(W, x1 + pad)
                near = m[yy0:yy1, xx0:xx1]
                drop[yy0:yy1, xx0:xx1] |= fg[yy0:yy1, xx0:xx1] & (cv2.dilate(near.astype(np.uint8), ker(pad)) > 0)
                cnt += 1
        big = still
    return drop, cnt


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    orig = np.asarray(Image.open(ORIG).convert("RGB")).copy()
    gt_rgb = np.asarray(Image.open(GT).convert("RGB")).copy()
    gt = np.any(gt_rgb < NONWHITE, axis=2)
    print(f"真值根面积 {100*gt.mean():.3f}%")

    # ---- 1. 扫收边 ----
    R.HALO_GROW = 0
    rows, best = [], None
    print(f"\n{'SHRINK':>7}{'IoU':>9}{'召回':>9}{'精确':>9}{'面积倍数':>10}")
    for k in [0, 1, 2, 3, 4]:
        R.SHRINK_PX = k
        root, _, _ = R.extract_root(orig, return_maps=False)
        inter = int((gt & root).sum())
        iou = inter / max(1, int((gt | root).sum()))
        rec = inter / max(1, int(gt.sum()))
        pre = inter / max(1, int(root.sum()))
        ratio = root.mean() / gt.mean()
        rows.append({"SHRINK_PX": k, "IoU": round(iou, 4), "recall": round(rec, 4),
                     "precision": round(pre, 4), "面积倍数": round(ratio, 3),
                     "root_pct": round(100 * root.mean(), 4)})
        print(f"{k:>7}{iou:>9.4f}{rec:>9.4f}{pre:>9.4f}{ratio:>10.3f}")
        if best is None or iou > best["IoU"]:
            best = rows[-1]
            best_root = root
    print(f"\n  最优 SHRINK_PX = {best['SHRINK_PX']}  IoU {best['IoU']:.4f}  "
          f"召回 {best['recall']:.4f}  精确 {best['precision']:.4f}  面积倍数 {best['面积倍数']:.3f}")

    with (OUT / "shrink_sweep.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    # ---- 2. 白底根图 ----
    KEEP = int(best["SHRINK_PX"])
    R.SHRINK_PX = KEEP
    root, _, _ = R.extract_root(orig, return_maps=False)
    out = orig.copy()
    out[~root] = 255
    pct_before = 100 * np.any(out < NONWHITE, axis=2).mean()

    # ---- 3. 清细壁闭合环 ----
    drop, n_loops = remove_loops(out)
    out2 = out.copy()
    out2[drop] = 255
    pct_after = 100 * np.any(out2 < NONWHITE, axis=2).mean()
    print(f"  细壁闭合环: {n_loops} 处, 剔除 {int(drop.sum()):,} px  "
          f"({pct_before:.3f}% -> {pct_after:.3f}%)")

    # ---- 4. 输出 ----
    Image.fromarray(out2).save(OUT / "7-003_final.jpg", "JPEG", quality=98,
                               subsampling=0, dpi=(600, 600))
    Image.fromarray(out2).save(OUT / "7-003_final.png", optimize=True)
    print(f"  -> {OUT/'7-003_final.jpg'}")
    print(f"  -> {OUT/'7-003_final.png'}")

    # 对比图：原图 | 你的真值 | 我的成品 | 误差
    def vis(base, mask, color=(0, 170, 0)):
        ov = base.astype(np.float32)
        ov[mask] = ov[mask] * 0.30 + np.array(color) * 0.70
        return ov.astype(np.uint8)

    final_mask = np.any(out2 < NONWHITE, axis=2)
    ov = orig.astype(np.float32)
    ov[gt & final_mask] = ov[gt & final_mask] * 0.35 + np.array([0, 170, 0]) * 0.65
    ov[~gt & final_mask] = ov[~gt & final_mask] * 0.2 + np.array([255, 40, 40]) * 0.8
    ov[gt & ~final_mask] = ov[gt & ~final_mask] * 0.2 + np.array([40, 90, 255]) * 0.8

    panels = [orig, np.stack([gt * 255] * 3, -1).astype(np.uint8),
              vis(orig, final_mask), ov.astype(np.uint8)]
    titles = ["① 原图", "② 你精修的真值", f"③ 我的成品（收边{KEEP}px + 清细线）",
              "④ 误差（红=误检 蓝=漏检）"]
    PW = 700
    tiles = [cv2.resize(p, (PW, int(p.shape[0] * PW / p.shape[1])), interpolation=cv2.INTER_AREA)
             for p in panels]
    th = max(t.shape[0] for t in tiles)
    row = np.hstack([np.pad(t, ((0, th - t.shape[0]), (0, 0), (0, 0))) for t in tiles])
    c = Image.new("RGB", (row.shape[1], row.shape[0] + 46), (255, 255, 255))
    c.paste(Image.fromarray(row), (0, 46))
    dr = ImageDraw.Draw(c)
    dr.text((8, 6), f"7-003 定稿   收边 {KEEP}px   面积 真值 {100*gt.mean():.3f}% / 我 {100*final_mask.mean():.3f}%"
                    f"   倍数 {final_mask.mean()/gt.mean():.2f}x", fill=(0, 0, 0), font=font(22))
    for i, t in enumerate(titles):
        dr.text((8 + i * PW, 28), t, fill=(150, 0, 0), font=font(17))
    c.save(OUT / "7-003_final_compare.png", optimize=True)
    print(f"  -> {OUT/'7-003_final_compare.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
