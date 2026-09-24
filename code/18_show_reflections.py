"""把检出的"水渍反光细线"画在原图上，供人工核对检测器是否抓对了目标。

用法:
    python 18_show_reflections.py --src=<图片目录> <文件名1> <文件名2> ...
产出: outputs/反光核对/<文件名>_rings.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

# --- 统一路径（见 paths.py；数据搬家只改那里）---
sys.path.insert(0, str(Path(__file__).parent))
import paths as P  # noqa: E402

PROJ = P.PROJ
OUT = PROJ / "outputs" / "反光核对"
FEAT_LONG = 1400        # 可被 --feat= 覆盖；反光线极细，太低会被压成亚像素而漏检
FLOOD_TOL = 20
DARK_DROP = 12
SAT_ROOT = 0.10
FRANGI_TUB = 0.20
HALO_GROW = 6
REFL_MIN_HOLE = 25
REFL_WALL_HALF_W = 2.0
REFL_OPEN_R = 2


def font(sz):
    for n in ("msyh.ttc", "simhei.ttf"):
        try:
            return ImageFont.truetype(n, sz)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


def analyze(path: Path):
    rgb = np.asarray(Image.open(path).convert("RGB"))
    H, W = rgb.shape[:2]
    sc = FEAT_LONG / max(H, W)
    small = cv2.resize(rgb, (int(W * sc), int(H * sc)), interpolation=cv2.INTER_AREA) if sc < 1 else rgb
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
    g = max(1, int(round(HALO_GROW * sc)))
    grown = cv2.dilate(core.astype(np.uint8),
                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * g + 1))) > 0
    core = core | (grown & fg & deep)

    r = REFL_OPEN_R
    ker_r = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    opened = cv2.morphologyEx(core.astype(np.uint8), cv2.MORPH_OPEN, ker_r) > 0
    holes_m = ndimage.binary_fill_holes(core) & ~core
    holes_o = ndimage.binary_fill_holes(opened) & ~opened
    refl_holes = holes_m & ~holes_o
    dt_r = cv2.distanceTransform(core.astype(np.uint8), cv2.DIST_L2, 5)
    line = np.zeros_like(core)
    nn, ll, ss, _ = cv2.connectedComponentsWithStats(refl_holes.astype(np.uint8), 8)
    dil = np.ones((2 * r + 3, 2 * r + 3), np.uint8)
    hits = []
    for j in range(1, nn):
        a = int(ss[j, cv2.CC_STAT_AREA])
        if a < max(4, int(REFL_MIN_HOLE * sc * sc)):
            continue
        seg = (cv2.dilate((ll == j).astype(np.uint8), dil) > 0) & core
        if seg.sum() < 8:
            continue
        w = float(dt_r[seg].max())
        if w <= REFL_WALL_HALF_W:
            line |= seg
            hits.append((int(ss[j, cv2.CC_STAT_LEFT]), int(ss[j, cv2.CC_STAT_TOP]),
                         int(ss[j, cv2.CC_STAT_WIDTH]), int(ss[j, cv2.CC_STAT_HEIGHT]), round(w, 2)))
    return small, core, line, hits, sc


def main() -> int:
    global FEAT_LONG, REFL_WALL_HALF_W, REFL_OPEN_R
    src = None
    names = []
    for a in sys.argv[1:]:
        if a.startswith("--src="):
            src = Path(a.split("=", 1)[1])
        elif a.startswith("--feat="):
            FEAT_LONG = int(a.split("=", 1)[1])
        elif a.startswith("--wall="):
            REFL_WALL_HALF_W = float(a.split("=", 1)[1])
        elif a.startswith("--openr="):
            REFL_OPEN_R = int(a.split("=", 1)[1])
        else:
            names.append(a)
    src = src or (PROJ / "processed" / "600dpi")
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"[核对] 特征尺度 {FEAT_LONG}px  壁厚上限 {REFL_WALL_HALF_W}  开运算半径 {REFL_OPEN_R}")
    files = [p for p in sorted(src.rglob("*"))
             if p.suffix.lower() in (".jpg", ".jpeg", ".tif", ".tiff", ".png") and p.stem in names]

    for p in files:
        small, core, line, hits, sc = analyze(p)
        ov = small.astype(np.float32).copy()
        ov[core] = ov[core] * 0.55 + np.array([0, 170, 0]) * 0.45      # 根=绿
        ov[line] = ov[line] * 0.15 + np.array([0, 90, 255]) * 0.85     # 反光=蓝
        big = cv2.resize(ov.astype(np.uint8), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        h, w = big.shape[:2]
        canvas = Image.new("RGB", (w + 24, h + 84), (255, 255, 255))
        canvas.paste(Image.fromarray(big), (12, 62))
        dr = ImageDraw.Draw(canvas)
        dr.text((12, 8), f"反光细线核对  {p.name}", fill=(0, 0, 0), font=font(26))
        dr.text((12, 38), f"绿=判为根   蓝=检出为水渍反光细线（{len(hits)} 处）   "
                          f"恢复系数 1/{1/sc:.2f}", fill=(150, 0, 0), font=font(20))
        canvas.save(OUT / f"{p.stem}_rings.png", optimize=True)
        print(f"  {p.name}: 蓝线 {len(hits)} 处  -> {OUT / (p.stem + '_rings.png')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
