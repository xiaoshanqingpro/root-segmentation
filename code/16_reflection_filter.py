"""水渍反光细线检测（用户 2026-09-22 反馈）。

用户反馈: "第二个数据集修改背景之后出现的此类细小闭合线条也是水渍带来的反射"

关键点: 这类反光线**有线性走向**，按 v1 标准「有管状走向就判根」会被误判成根。
        所以必须引入新判据。

物理与形态特征:
  - 反光线是**闭合**的（围出一个洞），根是树状结构，**不可能闭合**
  - 极细（半个宽度只有 1-2 px）
  - 常常浮在根旁边、与根系不连通

本脚本用"细 + 闭合（含洞）"两个条件识别，并统计在现有数据上的出现情况，
用来定阈值。产出诊断 CSV 与控制台汇总。
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# --- 统一路径（见 paths.py；数据搬家只改那里）---
sys.path.insert(0, str(Path(__file__).parent))
import paths as P  # noqa: E402

PROJ = P.PROJ
SRC = PROJ / "processed" / "600dpi"
OUT = PROJ / "outputs" / "反光线检测.csv"

FEAT_LONG = 1400
FLOOD_TOL = 20
DARK_DROP = 12
SAT_ROOT = 0.10
FRANGI_TUB = 0.20
HALO_GROW = 6
MIN_BLOB = 40

THIN_HALF_W = 2.5      # 半个宽度 <= 此值算"细"
MIN_HOLE = 30          # 洞面积 >= 此值才算"闭合"（滤掉噪声）


def root_and_features(path: Path):
    rgb = np.asarray(Image.open(path).convert("RGB"))
    H, W = rgb.shape[:2]
    sc = FEAT_LONG / max(H, W)
    small = cv2.resize(rgb, (int(W * sc), int(H * sc)), interpolation=cv2.INTER_AREA) if sc < 1 else rgb
    hh, ww = small.shape[:2]

    mask = np.zeros((hh + 2, ww + 2), np.uint8)
    flags = 4 | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | (255 << 8)
    img = np.ascontiguousarray(small)
    step = max(1, min(hh, ww) // 64)
    seeds = [(x, 0) for x in range(0, ww, step)] + [(x, hh - 1) for x in range(0, ww, step)] \
        + [(0, y) for y in range(0, hh, step)] + [(ww - 1, y) for y in range(0, hh, step)]
    # 第二数据集的图四边有扫描仪黑边：只从边界播种会把黑边填成背景，
    # 而真正要填的白纸反而不与边界连通 -> 整张图都成前景。所以再补一批"内部亮点"种子。
    gray0 = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    bright = gray0 > max(180, int(np.percentile(gray0, 80)))
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
    return core, depth, sc


def classify_components(core: np.ndarray):
    """返回每个连通块的 (是否反光线, 统计)"""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(core.astype(np.uint8), 8)
    out = []
    for j in range(1, n):
        area = int(stats[j, cv2.CC_STAT_AREA])
        if area < MIN_BLOB:
            continue
        x, y, w, h = stats[j, cv2.CC_STAT_LEFT], stats[j, cv2.CC_STAT_TOP], \
            stats[j, cv2.CC_STAT_WIDTH], stats[j, cv2.CC_STAT_HEIGHT]
        comp = (lab[y:y + h, x:x + w] == j)
        dt = cv2.distanceTransform(comp.astype(np.uint8), cv2.DIST_L2, 5)
        half_w = float(dt.max())
        # 洞 = 填洞后的面积 - 原面积
        filled = comp.copy()
        ff = filled.astype(np.uint8)
        cnts, _ = cv2.findContours(ff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        solid = np.zeros_like(ff)
        cv2.drawContours(solid, cnts, -1, 1, -1)
        hole = int(solid.sum() - comp.sum())
        out.append({"area": area, "half_w": round(half_w, 2), "hole": hole,
                    "w": int(w), "h": int(h)})
    return n - 1, out


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    src = SRC
    out_csv = OUT
    for a in sys.argv[1:]:
        if a.startswith("--src="):
            src = Path(a.split("=", 1)[1])
        elif a.startswith("--out="):
            out_csv = Path(a.split("=", 1)[1])
    files = sorted(p for p in src.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".tif", ".tiff", ".png"))
    if args:
        files = [p for p in files if p.stem in args]
    print(f"[反光检测] 源 {src}")
    print(f"[反光检测] {len(files)} 张  判据: 半宽<={THIN_HALF_W}px 且 洞面积>={MIN_HOLE} px2")
    rows = []
    for i, p in enumerate(files, 1):
        core, depth, sc = root_and_features(p)
        n_comp, comps = classify_components(core)
        refl = [c for c in comps if c["half_w"] <= THIN_HALF_W and c["hole"] >= MIN_HOLE]
        # 细但不闭合的（可能是真细根，也可能是断掉的线）
        thin_only = [c for c in comps if c["half_w"] <= THIN_HALF_W and c["hole"] < MIN_HOLE]
        px_all = sum(c["area"] for c in comps)
        px_refl = sum(c["area"] for c in refl)
        rows.append({
            "file": p.name, "rel_path": str(p.relative_to(src)),
            "root_px": px_all, "components": n_comp,
            "reflect_components": len(refl), "reflect_px": px_refl,
            "reflect_pct_of_root": round(100 * px_refl / max(1, px_all), 2),
            "thin_loopless_components": len(thin_only),
            "thin_loopless_px": sum(c["area"] for c in thin_only),
            "max_half_w": round(max((c["half_w"] for c in comps), default=0), 2),
        })
        if i % 20 == 0:
            print(f"  ... {i}/{len(files)}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    rc = np.array([r["reflect_components"] for r in rows])
    rp = np.array([r["reflect_pct_of_root"] for r in rows])
    tc = np.array([r["thin_loopless_components"] for r in rows])
    print(f"\n  含反光线的图: {int((rc>0).sum())}/{len(rows)}")
    print(f"  反光连通块总数: {int(rc.sum())}  单张最多 {int(rc.max())}")
    print(f"  反光像素占根像素: 中位 {np.median(rp):.2f}%  最大 {rp.max():.2f}%")
    print(f"  细但不闭合的块（需人工看是不是真细根）: 总数 {int(tc.sum())}，单张最多 {int(tc.max())}")
    top = sorted(rows, key=lambda r: -r["reflect_pct_of_root"])[:10]
    print("\n  反光占比最高的 10 张:")
    for r in top:
        print(f"    {r['file']:<18} 反光块 {r['reflect_components']:3d}  "
              f"{r['reflect_pct_of_root']:6.2f}%  细非闭合块 {r['thin_loopless_components']:3d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
