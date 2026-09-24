"""诊断白底根图的"残留"：保留下来的非白像素里，有多少其实是灰底/碎片而非根。

判据（供诊断，不是最终规则）:
  白底图里非白像素 = 被判为根而保留的像素
  其中真正像根的: 偏暗（灰度低）且偏褐（一定饱和度）
  其中像残留的:   偏亮（浅灰）或 饱和度极低（灰）或 面积很小的孤立连通块
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def report(p: Path) -> None:
    arr = np.asarray(Image.open(p).convert("RGB"))
    nonwhite = ~np.all(arr >= 250, axis=2)
    if nonwhite.sum() == 0:
        print(f"{p.name}: 全白")
        return
    rgb = arr[nonwhite].astype(np.float32)
    gray = (0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2])
    mx = rgb.max(axis=1)
    mn = rgb.min(axis=1)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1), 0)

    n = len(gray)
    light = gray > 180
    lowsat = sat < 0.06
    print(f"\n{p.name}  保留像素 {n}  ({100*nonwhite.mean():.3f}% of image)")
    print(f"  灰度: p10={np.percentile(gray,10):6.1f} p50={np.percentile(gray,50):6.1f} "
          f"p90={np.percentile(gray,90):6.1f}")
    print(f"  饱和度: p10={np.percentile(sat,10):.4f} p50={np.percentile(sat,50):.4f} "
          f"p90={np.percentile(sat,90):.4f}")
    print(f"  偏亮(灰度>180) {100*light.mean():5.1f}%   近乎无彩(饱和<0.06) {100*lowsat.mean():5.1f}%")
    print(f"  既偏亮又无彩（典型灰底） {100*(light&lowsat).mean():5.1f}%")

    # 连通块
    m = nonwhite.astype(np.uint8)
    nn, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    areas = np.array([stats[j, cv2.CC_STAT_AREA] for j in range(1, nn)])
    if len(areas):
        big = areas[areas >= 500]
        small = areas[areas < 500]
        print(f"  连通块 {nn-1} 个:  面积>=500 的 {len(big)} 个（占像素 {100*big.sum()/areas.sum():.1f}%）；"
              f"<500 的 {len(small)} 个（占 {100*small.sum()/max(1,areas.sum()):.2f}%）")


def main() -> int:
    paths = sys.argv[1:]
    for s in paths:
        p = Path(s)
        if p.is_dir():
            for q in sorted(p.rglob("*.jpg")):
                report(q)
        else:
            report(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
