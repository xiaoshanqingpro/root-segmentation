"""7-003 口径对比：原图 / 用户精修真值 / 我的自动结果（外扩0px 与 外扩1px）。

目的: 让用户直接看出"根径量到哪条边"的差别，并由他拍板。
产出: outputs/7-003三方对比/7-003_autovsgt.png
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
ORIG = P.RAW_D2_99 / "7-003.tif"
GT = P.TEST_MAN_99GAI / "7-003.tif"
OUT = PROJ / "outputs" / "7-003三方对比"
NONWHITE = 250


def font(sz):
    for n in ("msyh.ttc", "simhei.ttf"):
        try:
            return ImageFont.truetype(n, sz)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


def panel(img, title, box=None, width=680, zoom=False):
    if box:
        x0, y0, x1, y1 = box
        img = img[y0:y1, x0:x1]
    if zoom:
        Z = max(1, int(width / img.shape[1]))
        disp = cv2.resize(img, None, fx=Z, fy=Z, interpolation=cv2.INTER_NEAREST)
    else:
        disp = cv2.resize(img, (width, int(img.shape[0] * width / img.shape[1])),
                          interpolation=cv2.INTER_AREA)
    c = Image.new("RGB", (disp.shape[1], disp.shape[0] + 42), (255, 255, 255))
    c.paste(Image.fromarray(disp), (0, 42))
    ImageDraw.Draw(c).text((8, 8), title, fill=(0, 0, 0), font=font(21))
    return c


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    orig = np.asarray(Image.open(ORIG).convert("RGB")).copy()
    gt_rgb = np.asarray(Image.open(GT).convert("RGB")).copy()
    gt = np.any(gt_rgb < NONWHITE, axis=2)

    R.HALO_GROW = 0
    r0, _, _ = R.extract_root(orig, return_maps=False)
    R.HALO_GROW = 1
    r1, _, _ = R.extract_root(orig, return_maps=False)
    R.HALO_GROW = 6      # 复原默认
    print(f"真值 {100*gt.mean():.3f}%  外扩0 {100*r0.mean():.3f}%  外扩1 {100*r1.mean():.3f}%")

    def vis(root):
        ov = orig.astype(np.float32)
        ov[root] = ov[root] * 0.30 + np.array([0, 170, 0]) * 0.70
        return ov.astype(np.uint8)

    def err(root):
        ov = orig.astype(np.float32)
        ov[gt & root] = ov[gt & root] * 0.35 + np.array([0, 170, 0]) * 0.65
        ov[~gt & root] = ov[~gt & root] * 0.2 + np.array([255, 40, 40]) * 0.8   # 误检 红
        ov[gt & ~root] = ov[gt & ~root] * 0.2 + np.array([40, 90, 255]) * 0.8   # 漏检 蓝
        return ov.astype(np.uint8)

    # 找一个分叉/细线明显的区域放大
    H, W = orig.shape[:2]
    ys, xs = np.nonzero(gt)
    cy, cx = int(np.median(ys)), int(np.median(xs))
    BW, BH = 1100, 780
    x0 = int(np.clip(cx - BW // 2, 0, W - BW)); y0 = int(np.clip(cy - BH // 2, 0, H - BH))
    box = (x0, y0, x0 + BW, y0 + BH)

    top = [panel(orig, "① 原图"),
           panel(np.stack([gt * 255] * 3, -1).astype(np.uint8), "② 你精修的真值（最仔细的一张）"),
           panel(vis(r0), "③ 我的自动结果（贴根外扩 0px）"),
           panel(vis(r1), "④ 我的自动结果（贴根外扩 6px，旧默认）")]
    bot = [panel(orig, f"① 原图   局部({x0},{y0})", box, zoom=True),
           panel(np.stack([gt * 255] * 3, -1).astype(np.uint8), "② 你的真值", box, zoom=True),
           panel(err(r0), "③ 外扩0px  vs 真值（红=误检 蓝=漏检）", box, zoom=True),
           panel(err(r1), "④ 外扩6px  vs 真值（红=误检 蓝=漏检）", box, zoom=True)]
    h1 = max(t.height for t in top); h2 = max(t.height for t in bot)
    row1 = Image.new("RGB", (sum(t.width for t in top), h1), (255, 255, 255))
    x = 0
    for t in top:
        row1.paste(t, (x, 0)); x += t.width
    row2 = Image.new("RGB", (sum(t.width for t in bot), h2), (255, 255, 255))
    x = 0
    for t in bot:
        row2.paste(t, (x, 0)); x += t.width
    canvas = Image.new("RGB", (max(row1.width, row2.width), h1 + h2 + 20), (255, 255, 255))
    canvas.paste(row1, (0, 0)); canvas.paste(row2, (0, h1 + 20))
    o = OUT / "7-003_autovsgt.png"
    canvas.save(o, optimize=True)
    print(f"-> {o}  {canvas.size}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
