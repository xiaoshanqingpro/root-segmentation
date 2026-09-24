"""定向测量：9.9改\\44-001 里，"水泡轮廓线"与"真根"在灰度/饱和度上有没有可分性。

前面的拓扑判据（闭不闭合）在这种图上不可靠：细线在工作尺度上会碎成上千块。
所以退一步问一个更朴素的问题——它们是不是本来就**更浅**？
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# --- 统一路径（见 paths.py；数据搬家只改那里）---
sys.path.insert(0, str(Path(__file__).parent))
import paths as P  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
SRC = P.TEST_MAN_99GAI / "44-001.tif"
FEAT = 2600
FLOOD_TOL = 20
DARK_DROP = 12
SAT_ROOT = 0.10
FRANGI_TUB = 0.20


def main() -> int:
    rgb = np.asarray(Image.open(SRC).convert("RGB"))
    H, W = rgb.shape[:2]
    sc = FEAT / max(H, W)
    small = cv2.resize(rgb, (int(W * sc), int(H * sc)), interpolation=cv2.INTER_AREA)
    hh, ww = small.shape[:2]

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

    dt = cv2.distanceTransform(core.astype(np.uint8), cv2.DIST_L2, 5)
    print(f"特征图 {ww}x{hh}   前景 {100*fg.mean():.2f}%   core {100*core.mean():.3f}%")
    print(f"core 内 距离变换 分位: p50={np.percentile(dt[core],50):.2f} "
          f"p90={np.percentile(dt[core],90):.2f} max={dt[core].max():.2f}")
    for lo, hi, name in [(0, 1.5, "极细 (半宽<=1.5)"), (1.5, 2.5, "细 (1.5-2.5)"),
                         (2.5, 4, "中 (2.5-4)"), (4, 99, "粗 (>=4)")]:
        m = core & (dt > lo) & (dt <= hi)
        if m.sum() < 50:
            print(f"  {name:<18} 像素 {int(m.sum()):>7} 太少")
            continue
        print(f"  {name:<18} 像素 {int(m.sum()):>7}  "
              f"灰度均值 {gray[m].mean():6.1f}  深度均值 {depth[m].mean():6.1f}  "
              f"饱和度均值 {sat[m].mean():.4f}  Frangi均值 {fr[m].mean():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
