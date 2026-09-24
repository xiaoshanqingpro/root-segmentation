"""把机器修图导成 TIFF，规格与人工修图（7-003-man.tif）完全一致。

人工版规格（实测）: 7019x4962, RGB, 8bit/通道, 600 dpi, compression=raw(未压缩), 单帧
本脚本按同一规格导两档:
  7-003_auto.tif   收边1px（= 之前送扫的 7-003_auto.jpg 那版）
  7-003_auto2.tif  收边2px（面积最贴真值，供下一轮对照扫描）
"""
from __future__ import annotations

import os
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
ORIG = P.RAW_D2_99 / "7-003.tif"
OUTDIRS = [PROJ / "outputs" / "7-003三方对比", P.TEST]

# 细壁闭合环清理（水泡细线）
def remove_loops(rgb, nonwhite=250, min_hole=300, max_erode=3):
    import cv2
    from scipy import ndimage
    fg = np.any(rgb < nonwhite, axis=2)
    H, W = fg.shape
    ker = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    holes = ndimage.binary_fill_holes(fg) & ~fg
    drop = np.zeros_like(fg)
    if not holes.any():
        return drop
    nh, lh, sh, _ = cv2.connectedComponentsWithStats(holes.astype(np.uint8), 8)
    big = [(j, sh[j, cv2.CC_STAT_AREA]) for j in range(1, nh) if sh[j, cv2.CC_STAT_AREA] >= min_hole]
    for r in range(1, max_erode + 1):
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
                drop[yy0:yy1, xx0:xx1] |= fg[yy0:yy1, xx0:xx1] & (
                    cv2.dilate(near.astype(np.uint8), ker(pad)) > 0)
        big = still
    return drop


def main() -> int:
    orig = np.asarray(Image.open(ORIG).convert("RGB")).copy()
    print(f"原图 {orig.shape[1]}x{orig.shape[0]}")
    R.HALO_GROW = 0
    names = {1: "7-003_auto.tif", 2: "7-003_auto2.tif"}
    for k in (1, 2):
        R.SHRINK_PX = k
        root, _, _ = R.extract_root(orig, return_maps=False)
        out = orig.copy()
        out[~root] = 255
        out[remove_loops(out)] = 255
        m = np.any(out < 250, axis=2)
        img = Image.fromarray(out)
        for d in OUTDIRS:
            d.mkdir(parents=True, exist_ok=True)
            p = d / names[k]
            # 与 7-003-man.tif 同规格: 未压缩 raw TIFF, 8bit RGB, 600dpi
            img.save(p, format="TIFF", compression="raw", dpi=(600, 600))
            print(f"  收边{k}px  前景 {100*m.mean():.3f}%  ->  {p}  "
                  f"{os.path.getsize(p)/1024**2:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
