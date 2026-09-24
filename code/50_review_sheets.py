"""把预标注候选的预览拼成审阅总表，方便人工快速过目。"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
import paths as P  # noqa: E402

SRC = P.PROJ / "outputs" / "预标注候选" / "_preview"
OUT = P.PROJ / "outputs" / "预标注候选" / "_审阅总表"
COLS, ROWS = 6, 5
THUMB = 420


def font(sz):
    for n in ("msyh.ttc", "simhei.ttf"):
        try:
            return ImageFont.truetype(n, sz)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(SRC.glob("*_cand.png"))
    print(f"[总表] {len(files)} 张预览")
    per = COLS * ROWS
    n_sheet = (len(files) + per - 1) // per
    for s in range(n_sheet):
        chunk = files[s * per:(s + 1) * per]
        thumbs = []
        for p in chunk:
            im = Image.open(p).convert("RGB")
            # 裁掉预览顶部的标题条（保留图）
            im = im.crop((0, 54, im.width, im.height))
            im = im.resize((THUMB, max(1, int(im.height * THUMB / im.width))), Image.LANCZOS)
            thumbs.append((p.stem.replace("_cand", ""), im))
        th = max(t[1].height for t in thumbs)
        cell_h = th + 26
        sheet = Image.new("RGB", (COLS * THUMB, ROWS * cell_h), (245, 245, 245))
        dr = ImageDraw.Draw(sheet)
        f = font(18)
        for i, (name, im) in enumerate(thumbs):
            r, c = divmod(i, COLS)
            x, y = c * THUMB, r * cell_h
            sheet.paste(im, (x, y))
            dr.rectangle([x, y + th, x + THUMB, y + cell_h], fill=(255, 255, 255))
            dr.text((x + 6, y + th + 4), name, fill=(0, 0, 0), font=f)
        o = OUT / f"审阅_{s+1:02d}.png"
        sheet.save(o, optimize=True)
        print(f"  {o.name}  {sheet.size}  ({len(thumbs)} 张)")
    print(f"\n  总表 -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
