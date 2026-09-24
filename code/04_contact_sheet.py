"""阶段一 · 全量缩略图总表：用来一眼扫出非根杂物（尺子/标签/托盘边/污渍）与异常图。

产出: outputs/阶段一/contact_sheets/sheet_XX.png
每格下方标注 文件名，方便定位。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# --- 统一路径（见 paths.py；数据搬家只改那里）---
sys.path.insert(0, str(Path(__file__).parent))
import paths as P  # noqa: E402

IMAGE_DIR = P.RAW_D1
OUT_DIR = Path(r"D:\根系分割项目\outputs\阶段一\contact_sheets")
COLS, ROWS = 6, 4
THUMB_W = 420
PER_SHEET = COLS * ROWS


def font(size: int):
    for name in ("msyh.ttc", "simhei.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(IMAGE_DIR.rglob("*.jpg"))
    fnt = font(20)
    n_sheets = math.ceil(len(files) / PER_SHEET)
    print(f"[contact] {len(files)} 张 -> {n_sheets} 张总表", flush=True)

    for s in range(n_sheets):
        chunk = files[s * PER_SHEET:(s + 1) * PER_SHEET]
        thumbs = []
        for p in chunk:
            im = Image.open(p)
            im.draft("RGB", (THUMB_W, THUMB_W))
            im = im.convert("RGB")
            ratio = THUMB_W / im.width
            im = im.resize((THUMB_W, int(im.height * ratio)), Image.LANCZOS)
            thumbs.append((p.name, im))
        th = max(t[1].height for t in thumbs)
        cell_h = th + 30
        sheet = Image.new("RGB", (COLS * THUMB_W, ROWS * cell_h), (240, 240, 240))
        dr = ImageDraw.Draw(sheet)
        for i, (name, im) in enumerate(thumbs):
            r, c = divmod(i, COLS)
            x, y = c * THUMB_W, r * cell_h
            sheet.paste(im, (x, y))
            dr.rectangle([x, y + th, x + THUMB_W, y + cell_h], fill=(255, 255, 255))
            dr.text((x + 6, y + th + 5), name, fill=(0, 0, 0), font=fnt)
        out = OUT_DIR / f"sheet_{s + 1:02d}.png"
        sheet.save(out, optimize=True)
        print(f"  {out.name}  {sheet.size}  {[n for n, _ in thumbs][:6]} ...", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
