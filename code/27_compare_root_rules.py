"""判根规则回归测试：新旧规则对比 + 目视样例。

对若干张图同时跑旧规则与 rootseg_common 的新规则，
输出"灰底残留/碎片"指标对比，并生成放大对比图供人工复核。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
import rootseg_common as R  # noqa: E402

# --- 统一路径（见 paths.py；数据搬家只改那里）---
sys.path.insert(0, str(Path(__file__).parent))
import paths as P  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
PROJ = P.PROJ
OUT = PROJ / "outputs" / "判根规则对比"


def font(sz):
    for n in ("msyh.ttc", "simhei.ttf"):
        try:
            return ImageFont.truetype(n, sz)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


def old_rule(rgb: np.ndarray, feat_long: int = R.FEAT_LONG):
    """旧规则（复现）：管状判据没有暗度约束；无连通块级过滤；面积阈值被 sc² 抹平。"""
    H, W = rgb.shape[:2]
    sc = min(1.0, feat_long / max(H, W))
    small = cv2.resize(rgb, (int(W * sc), int(H * sc)), interpolation=cv2.INTER_AREA) if sc < 1 else rgb
    hh, ww = small.shape[:2]
    fg = ~R.flood_background(small)
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
    deep = depth > 12
    core = fg & deep & ((fr > 0.20) | (sat >= 0.10))
    g = max(1, int(round(6 * sc)))
    grown = cv2.dilate(core.astype(np.uint8),
                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * g + 1))) > 0
    core = core | (grown & fg & deep)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(core.astype(np.uint8), 8)
    keep = np.array([False] + [stats[j, cv2.CC_STAT_AREA] >= max(4, int(40 * sc * sc))
                               for j in range(1, n)])
    core = keep[lab]
    if sc < 1.0:
        core_full = cv2.resize(core.astype(np.uint8), (W, H),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
    else:
        core_full = core
    d = R.diagnose(gray, sat, fr, fg, core, sc)
    return core_full, d


def tile(rgb, root, title, box=None):
    if box:
        x0, y0, x1, y1 = box
        rgb = rgb[y0:y1, x0:x1]
        root = root[y0:y1, x0:x1]
    ov = rgb.astype(np.float32).copy()
    ov[root] = ov[root] * 0.35 + np.array([0, 190, 0]) * 0.65
    ov = ov.astype(np.uint8)
    Z = max(1, int(1000 / max(1, ov.shape[1])))
    ov = cv2.resize(ov, None, fx=Z, fy=Z, interpolation=cv2.INTER_NEAREST)
    h, w = ov.shape[:2]
    c = Image.new("RGB", (w, h + 34), (255, 255, 255))
    c.paste(Image.fromarray(ov), (0, 34))
    ImageDraw.Draw(c).text((6, 6), title, fill=(0, 0, 0), font=font(19))
    return c


def main() -> int:
    paths = [Path(a) for a in sys.argv[1:]]
    OUT.mkdir(parents=True, exist_ok=True)
    for p in paths:
        rgb = np.asarray(Image.open(p).convert("RGB"))
        new_root, fg, info = R.extract_root(rgb, return_maps=True)
        old_root, old_d = old_rule(rgb)
        new_d = info["diag_feat"]

        print(f"\n=== {p.name} ===")
        print(f"  旧规则: 保留 {old_d['root_pct']:6.3f}%  灰底残留 {old_d['graybg_pct']:5.1f}%  "
              f"碎片(<500px) {old_d['tiny_components']:5d} 个")
        print(f"  新规则: 保留 {new_d['root_pct']:6.3f}%  灰底残留 {new_d['graybg_pct']:5.1f}%  "
              f"碎片(<500px) {new_d['tiny_components']:5d} 个")
        print(f"  灰底残留下降 {old_d['graybg_pct']-new_d['graybg_pct']:+.1f} 个百分点，"
              f"根面积变化 {new_d['root_pct']-old_d['root_pct']:+.3f} 个百分点")

        # 找一个根密集区域做局部对比
        H, W = rgb.shape[:2]
        ys, xs = np.nonzero(new_root)
        if len(ys):
            cy, cx = int(np.median(ys)), int(np.median(xs))
            bw, bh = 1200, 900
            x0 = int(np.clip(cx - bw // 2, 0, W - bw)); y0 = int(np.clip(cy - bh // 2, 0, H - bh))
            box = (x0, y0, x0 + bw, y0 + bh)
        else:
            box = None
        a = tile(rgb, old_root, f"旧规则  {p.name}   灰底残留 {old_d['graybg_pct']:.1f}%", box)
        b = tile(rgb, new_root, f"新规则   灰底残留 {new_d['graybg_pct']:.1f}%", box)
        c = tile(rgb, np.zeros_like(new_root), "原图", box)
        row = Image.new("RGB", (a.width + b.width + c.width, max(a.height, b.height)), (255, 255, 255))
        row.paste(a, (0, 0)); row.paste(b, (a.width, 0)); row.paste(c, (a.width + b.width, 0))
        o = OUT / f"{p.stem}_compare.png"
        row.save(o, optimize=True)
        print(f"  对比图 -> {o}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
