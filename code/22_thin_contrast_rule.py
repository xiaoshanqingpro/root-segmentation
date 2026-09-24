"""【已作废 · 勿用】方案验证：用"细结构需要更高对比度"来剔除水泡轮廓线。

作废原因（用户 2026-09-22 复核 44-001_zoom.png 后反馈）:
    "有一些真根被误删了"、"中间那个红被剔除的好像全部剔到根中间了"
该规则按"离粗结构 >9px"划细结构，这个几何条件并不等同于水泡线——
脱离粗结构的真根（根尖、远离主根的侧根）同样落在里面，比阈值淡一点就被误删。
误删真根不可接受（根长直接损失），故放弃。保留此文件仅作过程记录。

当前策略: 水泡线**不自动删**，按《阴影标注判定标准 v2》在标注环节人工判定。

--- 以下为原始说明 ---

依据 (21_diag_contrast 的提示): 极细结构整体比根**更浅**
（但逐像素统计会被粗结构的边缘反锯齿像素污染，所以改成分块判断）。

规则:
  thick = 开运算(core, r=3)            -> 粗结构（根的主体）
  thin  = core 中不属于粗结构、且远离粗结构的部分
  root  = thick  ∪  { thin 且 深度 > DARK_STRONG }
  即: 细结构必须"够黑"才算根；水泡线淡 -> 被剔除；真根尖黑 -> 保留

产出: outputs/反光核对/<名>_thinrule.png （原图 / 原判根 / 应用规则后）
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
OUT = PROJ / "outputs" / "反光核对"
FEAT = 2600
FLOOD_TOL = 20
DARK_DROP = 12
SAT_ROOT = 0.10
FRANGI_TUB = 0.20
HALO_GROW = 6
OPEN_R = 3           # 粗结构定义半径
DARK_STRONG = 45     # 细结构要"够黑"的阈值（深度）


def font(sz):
    for n in ("msyh.ttc", "simhei.ttf"):
        try:
            return ImageFont.truetype(n, sz)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


def build(path: Path):
    rgb = np.asarray(Image.open(path).convert("RGB"))
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
    g = max(1, int(round(HALO_GROW * sc)))
    grown = cv2.dilate(core.astype(np.uint8),
                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * g + 1))) > 0
    core = core | (grown & fg & deep)

    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * OPEN_R + 1, 2 * OPEN_R + 1))
    thick = cv2.morphologyEx(core.astype(np.uint8), cv2.MORPH_OPEN, ker) > 0
    near_thick = cv2.dilate(thick.astype(np.uint8),
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * OPEN_R + 3,) * 2)) > 0
    thin = core & ~near_thick
    kept_thin = thin & (depth > DARK_STRONG)
    new = thick | kept_thin
    return small, core, new, thin, kept_thin


def main() -> int:
    src = None
    names = []
    for a in sys.argv[1:]:
        if a.startswith("--src="):
            src = Path(a.split("=", 1)[1])
        else:
            names.append(a)
    src = src or (PROJ / "processed" / "600dpi")
    OUT.mkdir(parents=True, exist_ok=True)
    files = [p for p in sorted(src.rglob("*"))
             if p.suffix.lower() in (".jpg", ".jpeg", ".tif", ".tiff", ".png") and p.stem in names]
    print(f"[细结构高对比规则] OPEN_R={OPEN_R} DARK_STRONG={DARK_STRONG} FEAT={FEAT}")
    for p in files:
        small, core, new, thin, kept = build(p)
        dropped = int((core & ~new).sum())
        print(f"  {p.name}: core {int(core.sum())}px -> 新 {int(new.sum())}px  "
              f"剔除 {dropped}px ({100*dropped/max(1,int(core.sum())):.1f}%)  "
              f"其中细结构 {int(thin.sum())}px, 保留的黑色细结构 {int(kept.sum())}px")
        panels = []
        for m, col in [(core, (0, 170, 0)), (new, (0, 170, 0))]:
            ov = small.astype(np.float32).copy()
            ov[m] = ov[m] * 0.45 + np.array(col) * 0.55
            if m is new:
                ov[core & ~new] = ov[core & ~new] * 0.35 + np.array([255, 60, 60]) * 0.65
            panels.append(ov.astype(np.uint8))
        panels.insert(0, small)
        big = [cv2.resize(x, None, fx=1.1, fy=1.1, interpolation=cv2.INTER_AREA) for x in panels]
        row = np.hstack(big)
        canvas = Image.new("RGB", (row.shape[1] + 20, row.shape[0] + 70), (255, 255, 255))
        canvas.paste(Image.fromarray(row), (10, 60))
        dr = ImageDraw.Draw(canvas)
        dr.text((10, 8), f"细结构高对比规则  {p.name}", fill=(0, 0, 0), font=font(24))
        dr.text((10, 36), "左=原图   中=原判根(全绿)   右=应用规则后(红=被剔除的淡细线)",
                fill=(150, 0, 0), font=font(19))
        canvas.save(OUT / f"{p.stem}_thinrule.png", optimize=True)
        print(f"    -> {OUT / (p.stem + '_thinrule.png')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
