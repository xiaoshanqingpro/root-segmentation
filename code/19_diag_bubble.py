"""针对 9.9改\\44-001 的定向诊断：那条大水泡闭合线，在掩膜层面到底长什么样？

目的: 找出一个**能真正分开"水泡轮廓线"与"真根"**的判据，而不是继续猜。
逐连通块报告: 面积、最大半宽、是否含洞、骨架端点/分叉数、是否与粗结构相连。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.morphology import skeletonize

# --- 统一路径（见 paths.py；数据搬家只改那里）---
sys.path.insert(0, str(Path(__file__).parent))
import paths as P  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
SRC = P.TEST_MAN_99GAI / "44-001.tif"
FEAT = int(sys.argv[1]) if len(sys.argv) > 1 else 2600
FLOOD_TOL = 20
DARK_DROP = 12
SAT_ROOT = 0.10
FRANGI_TUB = 0.20
HALO_GROW = 6


def main() -> int:
    rgb = np.asarray(Image.open(SRC).convert("RGB"))
    H, W = rgb.shape[:2]
    sc = FEAT / max(H, W)
    small = cv2.resize(rgb, (int(W * sc), int(H * sc)), interpolation=cv2.INTER_AREA)
    hh, ww = small.shape[:2]
    print(f"原图 {W}x{H}  ->  特征图 {ww}x{hh} (sc={sc:.3f})  "
          f"1 特征像素 = {1/sc:.2f} 原像素 = {25.4/600/sc:.4f} mm")

    mask = np.zeros((hh + 2, ww + 2), np.uint8)
    flags = 4 | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | (255 << 8)
    img = np.ascontiguousarray(small)
    step = max(1, min(hh, ww) // 64)
    gray0 = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    bright = gray0 > max(180, int(np.percentile(gray0, 80)))
    seeds = [(x, 0) for x in range(0, ww, step)] + [(x, hh - 1) for x in range(0, ww, step)] \
        + [(0, y) for y in range(0, hh, step)] + [(ww - 1, y) for y in range(0, hh, step)]
    for y in range(step, hh - step, max(1, hh // 24)):
        for x in range(step, ww - step, max(1, ww // 24)):
            if bright[y, x]:
                seeds.append((x, y))
    for (sx, sy) in seeds:
        if not mask[sy + 1, sx + 1]:
            cv2.floodFill(img, mask, (sx, sy), 255, (FLOOD_TOL,) * 3, (FLOOD_TOL,) * 3, flags)
    fg = ~(mask[1:-1, 1:-1] > 0)

    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1].astype(np.float32) / 255.0
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32)
    k = max(31, (int(min(hh, ww) * 0.05) // 2) * 2 + 1)
    illum = cv2.morphologyEx(gray.astype(np.uint8), cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(np.float32)
    depth = illum - gray
    corrected = np.clip(gray / np.maximum(illum, 1e-3) * float(illum.mean()), 0, 255)
    from skimage.filters import frangi as sk_frangi
    fr = sk_frangi((255.0 - corrected) / 255.0, sigmas=range(1, 6), black_ridges=False)
    fr = cv2.normalize(fr.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
    deep = depth > DARK_DROP
    core = fg & deep & ((fr > FRANGI_TUB) | (sat >= SAT_ROOT))
    g = max(1, int(round(HALO_GROW * sc)))
    grown = cv2.dilate(core.astype(np.uint8),
                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * g + 1))) > 0
    core = core | (grown & fg & deep)

    dt = cv2.distanceTransform(core.astype(np.uint8), cv2.DIST_L2, 5)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(core.astype(np.uint8), 8)
    print(f"\n连通块 {n-1} 个。按面积取前 15:")
    print(f"{'#':>3} {'面积':>7} {'最大半宽':>8} {'含洞':>5} {'宽高':>12}  {'骨架端点':>7} {'分叉':>5} {'mm粗':>7}")
    order = sorted(range(1, n), key=lambda j: -stats[j, cv2.CC_STAT_AREA])[:15]
    for j in order:
        x, y, w_, h_, a = stats[j]
        comp = lab[y:y + h_, x:x + w_] == j
        hw = float(dt[lab == j].max())
        filled = ndimage.binary_fill_holes(comp)
        hole = int(filled.sum() - comp.sum())
        sk = skeletonize(comp)
        nb = cv2.filter2D(sk.astype(np.uint8), -1, np.ones((3, 3), np.uint8),
                          borderType=cv2.BORDER_CONSTANT)
        ends = int((sk & (nb == 2)).sum())
        br = int((sk & (nb >= 4)).sum())
        print(f"{j:>3} {a:>7} {hw:>8.2f} {hole:>5} {str(w_)+'x'+str(h_):>12}  "
              f"{ends:>7} {br:>5} {2*hw*25.4/600/sc:>7.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
