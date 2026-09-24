"""把"被剔除的像素"标红叠回原人工图上，检查是否误删了根。

这是发布任何清理结果前必须做的检查：数字下降不等于删对了。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
import paths as P  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
CLEAN = P.PROJ / "outputs" / "边框清理" / "cleaned"
OUT = P.PROJ / "outputs" / "边框清理"
NONWHITE = 250


def font(sz):
    for n in ("msyh.ttc", "simhei.ttf"):
        try:
            return ImageFont.truetype(n, sz)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


def find_man(stem):
    for d in [P.TEST_MAN_99GAI, P.TEST_MAN]:
        for q in sorted(d.glob("*")):
            if "-man" in q.stem.lower() and \
                    q.stem.replace("-man", "").replace("-9.22", "") == stem:
                return q
    return None


def main() -> int:
    for stem in ["22-001", "44-001", "46-002"]:
        a = np.asarray(Image.open(find_man(stem)).convert("RGB"))
        b = np.asarray(Image.open(CLEAN / f"{stem}-man-clean.tif").convert("RGB"))
        removed = np.any(a < NONWHITE, axis=2) & np.all(b >= NONWHITE, axis=2)
        # 叠加：被删的标红
        ov = a.astype(np.float32)
        ov[removed] = ov[removed] * 0.2 + np.array([255, 30, 30]) * 0.8
        ov = ov.astype(np.uint8)

        # 全图缩略
        def sm(arr, w=1000):
            im = Image.fromarray(arr)
            return im.resize((w, int(im.height * w / im.width)), Image.LANCZOS)
        panels = [sm(a), sm(ov)]
        H = max(p.height for p in panels)
        row = Image.new("RGB", (sum(p.width for p in panels), H + 40), (255, 255, 255))
        x = 0
        for p in panels:
            row.paste(p, (x, 40)); x += p.width
        d = ImageDraw.Draw(row)
        d.text((8, 6), f"{stem}   红 = 被自动剔除的像素（检查是否含真根）", fill=(0, 0, 0),
               font=font(22))
        d.text((8, 22), "① 原人工图", fill=(150, 0, 0), font=font(17))
        d.text((sm(a).width + 8, 22), "② 剔除区域（红）", fill=(150, 0, 0), font=font(17))
        o = OUT / f"{stem}_removed_overlay.png"
        row.save(o, optimize=True)

        # 统计：被删的深色像素（根组织）有多少
        g = a.astype(np.float32).mean(axis=2)
        dark_rm = int((removed & (g < 150)).sum())
        print(f"  {stem}: 共删 {int(removed.sum()):,} px，其中深色(疑似根) {dark_rm:,} px "
              f"({100*dark_rm/max(1,int(removed.sum())):.1f}%)")
        print(f"    -> {o}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
