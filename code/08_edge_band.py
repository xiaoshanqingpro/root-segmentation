"""阶段一 · 四边伪影带宽度逐张测量 v3（决定裁剪量）。

前两版的坑:
  v1 "非背景占比 >4%" -> 根系密集的列也满足, 全库报 42mm(扫到上限)
  v2 "从最外缘连续 >30%" -> G86-001 是"白灰边 -> 粉色刻度尺 -> 纸面"的**不连续**结构, 直接漏检

v3 判据: 尺子/托盘边/纸边是**贯通整条画幅**的结构, 一根侧根永远不可能占满某列 50% 的高度。
  对每个列/行算 "贯通占比" = 该线上 饱和(>0.15) 或 极暗(比局部背景暗 40) 的像素比例。
  贯通占比 > 0.50 记为带结构。取外缘 25% 内**最外侧**的带结构位置作为伪影带外延。
  另外记录"带结构线的数量"以区分[一条连续尺子]与[零星根尖]。
  仅当带结构线数 >= 8 条(≈0.5mm 宽)才认定为伪影带, 避免把孤立根尖算进去。

产出: outputs/阶段一/edge_band.csv
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

IMAGE_DIR = P.RAW_D1
OUT_DIR = Path(r"D:\根系分割项目\outputs\阶段一")
EXCLUDE = {"树根004.jpg"}
SCAN_MAX = 0.25
THROUGH = 0.35
MIN_LINES = 3
DARK_DROP = 40
# 粉色刻度尺: RGB≈(246,226,244) -> R>G 且 B>G, 两个通道都高出 G 约 20。
# 褐色根系是 R>G>B, B-G<0 -> 不会被这条判据命中。这是把尺子和根分开的关键。
PINK_RG = 12
PINK_BG = 10


def measure(path: Path):
    with Image.open(path) as raw:
        full_w, full_h = raw.size
        dpi = float((raw.info.get("dpi") or (600, 600))[0])
    im = Image.open(path)
    im.draft("RGB", (1400, 1400))
    im = im.convert("RGB")
    s = 1400 / max(im.size)
    if s < 1:
        im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
    rgb = np.asarray(im)
    h, w = rgb.shape[:2]
    to_full = full_w / w

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1].astype(np.float32) / 255.0
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))
    illum = cv2.morphologyEx(gray.astype(np.uint8), cv2.MORPH_CLOSE, ker).astype(np.float32)
    chroma = rgb.astype(np.int16)
    r, g, b = chroma[..., 0], chroma[..., 1], chroma[..., 2]
    pink = ((r - g) > PINK_RG) & ((b - g) > PINK_BG)
    through = ((illum - gray) > DARK_DROP) | pink

    lim_x, lim_y = int(w * SCAN_MAX), int(h * SCAN_MAX)

    def extent(profile):
        """profile: 每条线的贯通占比(从外缘往内的顺序)。返回 (外延线数, 带结构线数)"""
        hot = np.nonzero(profile > THROUGH)[0]
        if len(hot) < MIN_LINES:
            return 0, 0
        return int(hot.max()) + 1, int(len(hot))

    left, nl = extent(through[:, :lim_x].mean(axis=0))
    right, nr = extent(through[:, ::-1][:, :lim_x].mean(axis=0))
    top, nt = extent(through[:lim_y, :].mean(axis=1))
    bottom, nb = extent(through[::-1, :][:lim_y, :].mean(axis=1))
    return {"dpi": dpi, "to_full": to_full,
            "left": left, "right": right, "top": top, "bottom": bottom,
            "lines": {"left": nl, "right": nr, "top": nt, "bottom": nb}}


def main() -> int:
    files = [p for p in sorted(IMAGE_DIR.rglob("*.jpg")) if p.name not in EXCLUDE]
    rows = []
    for i, p in enumerate(files, 1):
        m = measure(p)
        mmpp = 25.4 / m["dpi"]
        rec = {"file": p.name, "rel_path": str(p.relative_to(IMAGE_DIR.parent)), "dpi": m["dpi"]}
        for side in ["left", "right", "top", "bottom"]:
            full_px = int(round(m[side] * m["to_full"]))
            rec[f"{side}_px"] = full_px
            rec[f"{side}_mm"] = round(full_px * mmpp, 2)
        rec["max_mm"] = round(max(rec[f"{s}_mm"] for s in ["left", "right", "top", "bottom"]), 2)
        rows.append(rec)
        if i % 20 == 0:
            print(f"  ... {i}/{len(files)}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "edge_band.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    for sname in ["left", "right", "top", "bottom"]:
        v = np.array([r[f"{sname}_mm"] for r in rows])
        nz = int((v > 0).sum())
        print(f"  {sname:<7} 有伪影带 {nz:3d}/{len(rows)} 张 | 中位 {np.median(v):5.2f}mm  "
              f"p90 {np.percentile(v,90):5.2f}mm  最大 {v.max():6.2f}mm  "
              f"({rows[int(v.argmax())]['file']})")
    mx = np.array([r["max_mm"] for r in rows])
    print(f"\n  四边最大值 中位 {np.median(mx):.2f}mm  p90 {np.percentile(mx,90):.2f}mm  最大 {mx.max():.2f}mm")
    for t in [2.5, 4, 6, 8, 10]:
        print(f"  统一裁剪 {t:5.1f}mm 可覆盖 {float((mx <= t).mean()*100):6.1f}% 的图")
    bad = sorted([r for r in rows if r["max_mm"] > 2.5], key=lambda z: -z["max_mm"])
    print(f"\n  伪影带 >2.5mm 的图: {len(bad)} 张")
    for r in bad[:25]:
        print(f"    {r['file']:<16} L{r['left_mm']:6.2f} R{r['right_mm']:6.2f} "
              f"T{r['top_mm']:6.2f} B{r['bottom_mm']:6.2f} mm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
