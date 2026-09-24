"""给指定的几张样本做"原图 vs 机器候选"对照表。

用法: python 51_compare_pairs.py <样本名1> <样本名2> ...
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
import paths as P  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
CAND = P.PROJ / "outputs" / "预标注候选"
OUT = CAND / "_对照表"
ORIG_DIRS = [P.TEST_ORIG, P.RAW_D2_99, P.RAW_D2, P.RAW_D1]


def font(sz, bold=False):
    names = ("msyhbd.ttc", "msyh.ttc", "simhei.ttf") if bold else ("msyh.ttc", "simhei.ttf")
    for n in names:
        try:
            return ImageFont.truetype(n, sz)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


def find_orig(stem: str) -> Path | None:
    for d in ORIG_DIRS:
        if not d.exists():
            continue
        for p in d.rglob("*"):
            if p.suffix.lower() in (".tif", ".tiff", ".jpg", ".jpeg") and p.stem == stem:
                return p
    return None


def thumb(p: Path, w: int = 880) -> Image.Image:
    im = Image.open(p).convert("RGB")
    k = w / im.width
    return im.resize((w, max(1, int(im.height * k))), Image.LANCZOS)


def main() -> int:
    stems = sys.argv[1:]
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for stem in stems:
        o = find_orig(stem)
        c = CAND / f"{stem}-mac.tif"
        if o is None or not c.exists():
            print(f"  !! {stem}: 原图={o}  候选={c.exists()}")
            continue
        a = thumb(o)
        b = thumb(c)
        rows.append((stem, a, b))
        print(f"  {stem}: 原图 {o.parent.name}\\{o.name}")

    if not rows:
        return 1
    W = 880
    HDR = 40
    PAD = 16
    total_h = sum(max(a.height, b.height) + HDR + PAD for _, a, b in rows)
    sheet = Image.new("RGB", (W * 2 + PAD * 3, total_h + PAD), (255, 255, 255))
    dr = ImageDraw.Draw(sheet)
    y = PAD
    f = font(22, True)
    ft = font(19)
    for stem, a, b in rows:
        dr.text((PAD, y), f"{stem}", fill=(0, 0, 0), font=f)
        y += HDR - 6
        sheet.paste(a, (PAD, y))
        sheet.paste(b, (W + PAD * 2, y))
        dr.text((PAD + 4, y + 4), "原图", fill=(150, 0, 0), font=ft)
        dr.text((W + PAD * 2 + 4, y + 4), "机器候选", fill=(0, 110, 0), font=ft)
        yy = y + max(a.height, b.height)
        dr.line([(0, yy + PAD // 2), (sheet.width, yy + PAD // 2)], fill=(200, 200, 200), width=2)
        y = yy + PAD

    o = OUT / "对照表.png"
    sheet.save(o, optimize=True)
    print(f"\n  -> {o}  {sheet.size}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
