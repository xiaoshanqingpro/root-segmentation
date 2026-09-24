"""局部放大对比：水泡闭合线所在区域，应用"细结构高对比"规则前后。

用法: python 23_zoom_compare.py <图片路径> <x0> <y0> <x1> <y1> [FEAT]
坐标是**特征图**坐标（FEAT 长边下的像素坐标）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
Image.MAX_IMAGE_PIXELS = None
PROJ = P.PROJ
OUT = PROJ / "outputs" / "反光核对"

from importlib import util as _il  # noqa: E402

# --- 统一路径（见 paths.py；数据搬家只改那里）---
sys.path.insert(0, str(Path(__file__).parent))
import paths as P  # noqa: E402

spec = _il.spec_from_file_location("tcr", str(Path(__file__).parent / "22_thin_contrast_rule.py"))
tcr = _il.module_from_spec(spec)
spec.loader.exec_module(tcr)


def font(sz):
    for n in ("msyh.ttc", "simhei.ttf"):
        try:
            return ImageFont.truetype(n, sz)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


def main() -> int:
    path = Path(sys.argv[1])
    x0, y0, x1, y1 = (int(v) for v in sys.argv[2:6])
    if len(sys.argv) > 6:
        tcr.FEAT = int(sys.argv[6])
    small, core, new, thin, kept = tcr.build(path)
    sl = (slice(y0, y1), slice(x0, x1))
    rgb_c, core_c, new_c = small[sl], core[sl], new[sl]
    dropped = core_c & ~new_c

    tiles = []
    ov1 = rgb_c.astype(np.float32).copy()
    ov1[core_c] = ov1[core_c] * 0.45 + np.array([0, 170, 0]) * 0.55
    tiles.append(("原判根（全绿）", ov1))
    ov2 = rgb_c.astype(np.float32).copy()
    ov2[new_c] = ov2[new_c] * 0.45 + np.array([0, 170, 0]) * 0.55
    ov2[dropped] = ov2[dropped] * 0.25 + np.array([255, 40, 40]) * 0.75
    tiles.append(("应用规则后（红=剔除）", ov2))
    tiles.append(("原图", rgb_c.astype(np.float32)))

    Z = max(1, int(1100 / max(1, rgb_c.shape[1])))
    outs = [cv2.resize(t[1].astype(np.uint8), None, fx=Z, fy=Z,
                       interpolation=cv2.INTER_NEAREST) for t in tiles]
    row = np.hstack(outs)
    canvas = Image.new("RGB", (row.shape[1] + 20, row.shape[0] + 70), (255, 255, 255))
    canvas.paste(Image.fromarray(row), (10, 60))
    dr = ImageDraw.Draw(canvas)
    dr.text((10, 8), f"水泡闭合线局部对比  {path.name}  区域({x0},{y0})-({x1},{y1})  放大{Z}x",
            fill=(0, 0, 0), font=font(22))
    seg = row.shape[1] // 3
    for i, (t, _) in enumerate(tiles):
        dr.text((12 + i * seg, 36), t, fill=(150, 0, 0), font=font(19))
    o = OUT / f"{path.stem}_zoom.png"
    canvas.save(o, optimize=True)
    print(f"区域内核像素 {int(core_c.sum())} -> 新 {int(new_c.sum())}，剔除 {int(dropped.sum())}")
    print(f"-> {o}  ({canvas.size})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
