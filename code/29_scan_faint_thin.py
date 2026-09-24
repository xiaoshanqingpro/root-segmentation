"""扫描白底根图输出里的"细且淡"结构，找出疑似水泡细线的图。

不预设"闭合"，先按最朴素的形态特征扫:
  细 = 距离变换很小（结构很薄）
  淡 = 偏离纯白的程度很小（对比度低）
  且 远离粗结构（不是根的边缘，是游离的细线）
按"细淡像素数"给每张图排序，再渲染 top 图的局部放大，供人工确认。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# --- 统一路径（见 paths.py；数据搬家只改那里）---
sys.path.insert(0, str(Path(__file__).parent))
import paths as P  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
PROJ = P.PROJ
OUT = PROJ / "outputs" / "细淡结构排查"

NONWHITE = 252          # 宽松阈值: 只要偏离纯白就算
THIN_DT = 1.5           # 距离变换 <= 此值算"细"
FAR_FROM_THICK = 6      # 离粗结构(半径>=3)超过此距离才算"游离"
FAINT = 0.30            # (255-灰度)/255 <= 此值算"淡"


def font(sz):
    for n in ("msyh.ttc", "simhei.ttf"):
        try:
            return ImageFont.truetype(n, sz)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


def scan(p: Path):
    arr = np.asarray(Image.open(p).convert("L"))
    fg = arr < NONWHITE
    if fg.sum() < 100:
        return 0, 0.0, None, None
    dt = cv2.distanceTransform(fg.astype(np.uint8), cv2.DIST_L2, 5)
    thick = dt >= 3.0
    if thick.any():
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * FAR_FROM_THICK + 1,) * 2)
        near_thick = cv2.dilate(thick.astype(np.uint8), k) > 0
    else:
        near_thick = np.zeros_like(fg)
    faint = (255.0 - arr.astype(np.float32)) / 255.0 <= FAINT
    # 细 & 淡 & 游离
    tfd = fg & (dt <= THIN_DT) & faint & ~near_thick
    n = int(tfd.sum())
    return n, 100.0 * n / max(1, int(fg.sum())), tfd, arr


def zoom(im_gray: np.ndarray, mask: np.ndarray, box, path: Path, title: str):
    x0, y0, x1, y1 = box
    a = im_gray[y0:y1, x0:x1]
    m = mask[y0:y1, x0:x1]
    rgb = np.stack([a] * 3, -1).astype(np.uint8)
    ov = rgb.astype(np.float32)
    ov[m] = ov[m] * 0.35 + np.array([255, 40, 40]) * 0.65
    Z = max(1, int(1100 / max(1, rgb.shape[1])))
    big = cv2.resize(ov.astype(np.uint8), None, fx=Z, fy=Z, interpolation=cv2.INTER_NEAREST)
    c = Image.new("RGB", (big.shape[1] + 16, big.shape[0] + 62), (255, 255, 255))
    c.paste(Image.fromarray(big), (8, 52))
    dr = ImageDraw.Draw(c)
    dr.text((8, 6), title, fill=(0, 0, 0), font=font(22))
    dr.text((8, 30), "红=细且淡且游离的像素（疑似水泡细线）", fill=(160, 0, 0), font=font(18))
    path.parent.mkdir(parents=True, exist_ok=True)
    c.save(path, optimize=True)
    return c.size


def main() -> int:
    src = PROJ / (sys.argv[1] if len(sys.argv) > 1 else "白底根图_第二数据集")
    top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(src.rglob("*.jpg"))
    print(f"[细淡结构排查] {src}  {len(files)} 张")
    res = []
    for p in files:
        n, pct, tfd, arr = scan(p)
        res.append((n, pct, p))
    res.sort(key=lambda z: -z[0])
    print(f"\n  {'文件':<24}{'细淡游离像素':>12}{'占前景比':>10}")
    for n, pct, p in res[:top_n + 10]:
        print(f"  {p.name:<24}{n:>12}{pct:>9.2f}%")
    print(f"\n  全库合计 细淡游离像素 {sum(r[0] for r in res)}，"
          f"有此类像素的图 {sum(1 for r in res if r[0] > 0)}/{len(res)}")
    for n, pct, p in res[:top_n]:
        _, _, tfd, arr = scan(p)
        ys, xs = np.nonzero(tfd)
        H, W = arr.shape
        bw, bh = min(1400, W), min(1000, H)
        cy, cx = int(np.median(ys)), int(np.median(xs))
        x0 = int(np.clip(cx - bw // 2, 0, W - bw)); y0 = int(np.clip(cy - bh // 2, 0, H - bh))
        o = OUT / f"{p.stem}_zoom.png"
        sz = zoom(arr, tfd, (x0, y0, x0 + bw, y0 + bh), o,
                  f"{p.name}   细淡游离像素 {n} ({pct:.2f}%)")
        print(f"  -> {o}  {sz}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
