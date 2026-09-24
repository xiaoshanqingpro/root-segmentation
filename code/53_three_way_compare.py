"""三版本对照表：同一批样本的 人工 / CV / U-Net 三种结果并排。

用于回答：经典规则管线(CV) 与 U-Net 到底谁更接近人工结果。
产出: outputs/三版本盘点/三版对照_XX.png（每张 4 个样本，每个样本一行三列）
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
import paths as P  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
OUT = P.PROJ / "outputs" / "三版本盘点"
OUT.mkdir(parents=True, exist_ok=True)
TW = 600           # 每格缩略宽度
PER_SHEET = 4      # 每张表几个样本


def font(sz, bold=False):
    names = ("msyhbd.ttc", "msyh.ttc", "simhei.ttf") if bold else ("msyh.ttc", "simhei.ttf")
    for n in names:
        try:
            return ImageFont.truetype(n, sz)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


def index(dirp: Path, suffixes) -> dict[str, Path]:
    m = {}
    if not dirp.exists():
        return m
    for p in dirp.rglob("*"):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".tif", ".tiff", ".png"):
            continue
        s = p.stem
        for suf in suffixes:
            if suf == "":
                m[s] = p
                break
            if s.endswith(suf):
                m[s[: -len(suf)]] = p
                break
    return m


def thumb(p: Path, w=TW) -> Image.Image:
    im = Image.open(p).convert("RGB")
    k = w / im.width
    return im.resize((w, max(1, int(im.height * k))), Image.LANCZOS)


def main() -> int:
    cand = P.PROJ / "outputs" / "预标注候选"
    unet = {k: v for k, v in index(cand, ["-unet", "-mac"]).items() if v.parent == cand}
    man = {}
    for d in [P.TEST_MAN_99GAI, P.TEST_MAN, P.PROJ / "人工修复mac图"]:
        man.update(index(d, ["-unet-man", "-man"]))
    cv = {}
    for d in [P.WHITE_D1, P.WHITE_D2]:
        cv.update(index(d, [""]))

    ids = [k for k in sorted(set(unet) & set(man) & set(cv))]
    print(f"三版俱全的样本 {len(ids)} 个")

    for s in range((len(ids) + PER_SHEET - 1) // PER_SHEET):
        chunk = ids[s * PER_SHEET:(s + 1) * PER_SHEET]
        rows = []
        for k in chunk:
            rows.append((k, thumb(man[k]), thumb(cv[k]), thumb(unet[k])))
        HDR, PAD = 36, 14
        row_h = max(max(r[1].height, r[2].height, r[3].height) for r in rows) + HDR + PAD
        sheet = Image.new("RGB", (TW * 3 + PAD * 4, row_h * len(rows) + PAD), (255, 255, 255))
        dr = ImageDraw.Draw(sheet)
        f, ft = font(22, True), font(18)
        y = PAD
        for k, a, b, c in rows:
            dr.text((PAD, y), k, fill=(0, 0, 0), font=f)
            y += HDR - 6
            sheet.paste(a, (PAD, y))
            sheet.paste(b, (TW + PAD * 2, y))
            sheet.paste(c, (TW * 2 + PAD * 3, y))
            dr.text((PAD + 4, y + 4), "人工", fill=(150, 0, 0), font=ft)
            dr.text((TW + PAD * 2 + 4, y + 4), "CV（经典规则）", fill=(0, 0, 160), font=ft)
            dr.text((TW * 2 + PAD * 3 + 4, y + 4), "U-Net", fill=(0, 120, 0), font=ft)
            yy = y + max(a.height, b.height, c.height)
            dr.line([(0, yy + PAD // 2), (sheet.width, yy + PAD // 2)], fill=(210, 210, 210), width=2)
            y = yy + PAD
        o = OUT / f"三版对照_{s+1:02d}.png"
        sheet.save(o, optimize=True)
        print(f"  {o.name}  {sheet.size}  {chunk}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
