"""水渍反光细线检测 v2：从"洞"反推包围它的细线。

v1 的失败: 按连通块找"细 + 闭合"。反光线一旦与根相连就并成一个大块，
           整块的半宽由根决定 -> 判不出来。实测两个数据集都报 0。

v2 的判据（与是否连在根上无关）:
  1. 在根掩膜里找**洞**（fill_holes - mask）。反光线围出的就是洞；根是树状结构，天然没有洞
  2. 取洞外沿 1px 的掩膜壳（ring）
  3. 若这层壳**整体很细**（距离变换的 p90 <= THIN_W），则围出该洞的是细线 -> 判为反光
  4. 把该壳从根掩膜里剔除

物理依据: 反光是水面/玻璃面形成的"影子"，是闭合的、极细的、平滑的；
         根不会闭合成环，且根再细也有实心的厚度。

产出: CSV + 控制台汇总
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

# --- 统一路径（见 paths.py；数据搬家只改那里）---
sys.path.insert(0, str(Path(__file__).parent))
import paths as P  # noqa: E402

PROJ = P.PROJ
FEAT_LONG = 1400
FLOOD_TOL = 20
DARK_DROP = 12
SAT_ROOT = 0.10
FRANGI_TUB = 0.20
HALO_GROW = 6
MIN_BLOB = 40

THIN_W = 2.2        # 围出该洞的"壁"的半个厚度上限（距离变换值）
MIN_HOLE = 25       # 洞面积下限（特征尺度像素）
SHELL = 3           # 兼容旧字段


def root_mask(path: Path):
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
    return core, sat, depth


def find_rings(mask: np.ndarray, r: int = 2):
    """用开运算测"壁厚"来识别反光细线围出的环。

    为什么不用"洞外沿 1px 壳的 dt"——那层壳天生就只有 1px，dt 恒等于 1.41，判据会退化成
    "所有洞都是反光"（实测 30/30 全中，是假阳性）。

    正确做法:
      细壁围出的洞，在**开运算**（结构元半径 r）之后会连同细壁一起消失（洞变成外部区域）；
      厚根围出的洞（根之间的缝隙）则纹丝不动。
      于是: 反光洞 = 在原掩膜里是洞，但在开运算结果里不再是洞。
      反光线 = 紧邻这些洞的掩膜像素。
    """
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    opened = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, ker) > 0
    filled_m = ndimage.binary_fill_holes(mask)
    filled_o = ndimage.binary_fill_holes(opened)
    holes_m = filled_m & ~mask
    holes_o = filled_o & ~opened
    refl_holes = holes_m & ~holes_o

    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(refl_holes.astype(np.uint8), 8)
    keep = np.zeros((n + 1, n + 1), np.uint8)  # 占位，下面重建
    keep = np.zeros_like(refl_holes)
    info = []
    dil = np.ones((2 * r + 3, 2 * r + 3), np.uint8)
    line = np.zeros_like(mask)
    for j in range(1, n):
        area = int(stats[j, cv2.CC_STAT_AREA])
        if area < MIN_HOLE:
            continue
        h = lab == j
        seg = (cv2.dilate(h.astype(np.uint8), dil) > 0) & mask
        if seg.sum() < 8:
            continue
        wall = float(dt[seg].max())
        isr = wall <= THIN_W
        info.append({"hole_area": area, "line_px": int(seg.sum()),
                     "wall_half_w": round(wall, 2), "is_reflection": bool(isr)})
        if isr:
            line |= seg
    return line, info


def main() -> int:
    src = None
    out_csv = None
    names = []
    for a in sys.argv[1:]:
        if a.startswith("--src="):
            src = Path(a.split("=", 1)[1])
        elif a.startswith("--out="):
            out_csv = Path(a.split("=", 1)[1])
        else:
            names.append(a)
    src = src or (PROJ / "processed" / "600dpi")
    out_csv = out_csv or (PROJ / "outputs" / "反光细线检测.csv")
    files = sorted(p for p in src.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".tif", ".tiff", ".png"))
    if names:
        files = [p for p in files if p.stem in names]
    print(f"[反光检测v2] 源 {src}")
    print(f"[反光检测v2] {len(files)} 张  壳 dt p90 <= {THIN_W} 判为反光细线，洞面积 >= {MIN_HOLE}")

    rows = []
    for i, p in enumerate(files, 1):
        mask, sat, depth = root_mask(p)
        remove, info = find_rings(mask)
        refl = [z for z in info if z["is_reflection"]]
        rows.append({
            "file": p.name, "rel_path": str(p.relative_to(src)),
            "root_px": int(mask.sum()),
            "holes": len(info), "reflection_rings": len(refl),
            "reflection_px": int(remove.sum()),
            "reflection_pct_of_root": round(100 * remove.sum() / max(1, mask.sum()), 2),
            "max_shell_dt_p90": round(max((z["wall_half_w"] for z in info), default=0), 2),
        })
        if refl:
            print(f"  [{i}/{len(files)}] {p.name}: 洞 {len(info)} 个, 其中 {len(refl)} 个是细线环"
                  f"  (壁厚 {min(z['wall_half_w'] for z in refl):.2f})", flush=True)
        elif i % 25 == 0:
            print(f"  ... {i}/{len(files)}", flush=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    rg = np.array([r["reflection_rings"] for r in rows])
    print(f"\n  含反光细线的图: {int((rg>0).sum())}/{len(rows)}   环总数 {int(rg.sum())}")
    rp = np.array([r["reflection_pct_of_root"] for r in rows])
    print(f"  反光像素占根像素: 中位 {np.median(rp):.2f}%  最大 {rp.max():.2f}%")
    print(f"  清单: {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
