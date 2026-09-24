"""阶段一 · 关键决策支持：阴影与根的空间关系。

要回答的问题（任务书里的关键决策点）:
    阴影到底是"根的软边/厚度晕圈"（贴着根，属于根的延伸 → 倾向留在根类或单设第 4 类）
    还是"纸面褶皱/水渍/照明不均"（远离根，独立存在 → 三类足够）？

做法: 对每张图，先得到根核心区与阴影嫌疑区，再用距离变换统计
      每个阴影连通块到最近根像素的距离，以及阴影像素的距离分布。

产出: outputs/阶段一/shadow_root_overlap.csv + 控制台直方图
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
WORK_LONG_SIDE = 1100
EXCLUDE = {"树根004.jpg"}  # 透射背光扫描件，成像原理不同，单独讨论


def load_gray_sat(path: Path):
    im = Image.open(path)
    im.draft("RGB", (WORK_LONG_SIDE, WORK_LONG_SIDE))
    im = im.convert("RGB")
    s = WORK_LONG_SIDE / max(im.size)
    if s < 1:
        im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
    rgb = np.asarray(im)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32), hsv[..., 1].astype(np.float32) / 255.0


def layers(gray: np.ndarray, sat: np.ndarray):
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))
    illum = cv2.morphologyEx(gray.astype(np.uint8), cv2.MORPH_CLOSE, ker).astype(np.float32)
    corrected = np.clip(gray / np.maximum(illum, 1e-3) * float(illum.mean()), 0, 255)
    inv = (255.0 - corrected) / 255.0
    try:
        from skimage.filters import frangi as sk_frangi
        fr = sk_frangi(inv, sigmas=range(1, 7), black_ridges=False)
        fr = cv2.normalize(fr.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
    except Exception:  # noqa: BLE001
        bh = cv2.morphologyEx(corrected.astype(np.uint8), cv2.MORPH_BLACKHAT,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        fr = cv2.normalize(bh.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
    depth = illum - gray
    deep = depth > 14
    root = deep & ((fr > 0.22) | (sat >= 0.12))
    shadow = deep & (sat < 0.12) & ~(fr > 0.22)
    clean = cv2.morphologyEx(root.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    shclean = cv2.morphologyEx(shadow.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    return clean, shclean


def main() -> int:
    files = [p for p in sorted(IMAGE_DIR.rglob("*.jpg")) if p.name not in EXCLUDE]
    rows, all_dist = [], []
    for i, p in enumerate(files, 1):
        gray, sat = load_gray_sat(p)
        root, shadow = layers(gray, sat)
        if root.sum() == 0 or shadow.sum() == 0:
            continue
        dist = cv2.distanceTransform((~root).astype(np.uint8), cv2.DIST_L2, 5)
        d_shadow = dist[shadow]
        all_dist.append(d_shadow)
        near = float((d_shadow <= 6).mean() * 100)      # 6px 内 -> 视为根的软边/晕圈
        mid = float(((d_shadow > 6) & (d_shadow <= 25)).mean() * 100)
        far = float((d_shadow > 25).mean() * 100)       # 远离根 -> 独立的纸面/水渍
        rows.append({
            "file": p.name,
            "rel_path": str(p.relative_to(IMAGE_DIR.parent)),
            "root_pct": round(float(root.mean() * 100), 3),
            "shadow_pct": round(float(shadow.mean() * 100), 3),
            "shadow_dist_p50": round(float(np.median(d_shadow)), 1),
            "shadow_near_root_pct(<=6px)": round(near, 1),
            "shadow_mid_pct(6-25px)": round(mid, 1),
            "shadow_far_pct(>25px)": round(far, 1),
        })
        if i % 20 == 0:
            print(f"  ... {i}/{len(files)}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "shadow_root_overlap.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    cat = np.concatenate(all_dist)
    print(f"\n=== 阴影像素到最近根像素的距离分布（全部 {len(rows)} 张，共 {cat.size} 像素）===")
    edges = [0, 3, 6, 12, 25, 50, 100, 200, 10 ** 9]
    names = ["0-3px", "3-6px", "6-12px", "12-25px", "25-50px", "50-100px", "100-200px", ">200px"]
    prev = 0
    for e, nm in zip(edges[1:], names):
        frac = float(((cat > prev) & (cat <= e)).mean() * 100)
        bar = "#" * int(frac / 2)
        print(f"  {nm:>10}  {frac:6.2f}%  {bar}")
        prev = e
    print(f"\n  中位数 {np.median(cat):.1f}px   均值 {cat.mean():.1f}px")
    near_all = float((cat <= 6).mean() * 100)
    far_all = float((cat > 25).mean() * 100)
    print(f"  <=6px(贴根晕圈) {near_all:.1f}%     >25px(独立阴影) {far_all:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
