"""7-003 三方对比图：原图 / 用户精修图 / 我的自动图 + 两次 WinRHIZO 扫描截图 + 扫描数据。

数据来源: C:\\Users\\LIHAOYANG\\Desktop\\测试\\测试数据.TXT
  （RHIZO 2016a 输出，两条记录: 7003man 与 7003auto）
原图无法扫描（背景不干净），故该列只放原图、扫描栏标注"无法扫描"。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import sys

# --- 统一路径（见 paths.py；数据搬家只改那里）---
sys.path.insert(0, str(Path(__file__).parent))
import paths as P  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
SRC = P.TEST
OUT = Path(r"D:\根系分割项目\outputs\7-003三方对比")

# 从 测试数据.TXT 解析出来的关键指标
DATA = [
    ("根总长 Length (cm)",         "221.15",   "245.42",   "+11.0%"),
    ("投影面积 ProjArea (cm²)",    "12.7997",  "14.9536",  "+16.8%"),
    ("表面积 SurfArea (cm²)",      "40.2113",  "46.9781",  "+16.8%"),
    ("平均直径 AvgDiam (mm)",      "0.5788",   "0.6093",   "+5.3%"),
    ("根体积 Volume (cm³)",        "0.582",    "0.716",    "+23.0%"),
    ("根尖数 Tips",                "408",      "629",      "+54.2%"),
    ("分叉数 Forks",               "626",      "848",      "+35.5%"),
    ("交叉数 Crossings",           "15",       "6",        "-60.0%"),
    ("分析区面积 (cm²)",           "425.83",   "446.71",   "+4.9%"),
]


def font(sz, bold=False):
    names = ("msyhbd.ttc", "msyh.ttc", "simhei.ttf") if bold else ("msyh.ttc", "simhei.ttf")
    for n in names:
        try:
            return ImageFont.truetype(n, sz)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


def load_fit(p: Path, w: int, h: int) -> Image.Image:
    im = Image.open(p).convert("RGB")
    k = min(w / im.width, h / im.height)
    im = im.resize((int(im.width * k), int(im.height * k)), Image.LANCZOS)
    bg = Image.new("RGB", (w, h), (255, 255, 255))
    bg.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
    return bg


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    CW, IH, SH = 900, 640, 560          # 列宽 / 图像高 / 截图高

    cols = [
        ("① 原图", SRC / "7-003.tif", None,
         "原图无法扫描\n（背景不干净，WinRHIZO 无法分析）"),
        ("② 人工修图  (7-003-man.tif)", SRC / "7-003-man.tif", SRC / "7003-man-scan.png",
         None),
        ("③ 机器修图  (7-003_auto.jpg)", SRC / "7-003_auto.jpg", SRC / "7003-auto-scan.png",
         None),
    ]

    imgs, scans = [], []
    for _, ip, sp, ph in cols:
        imgs.append(load_fit(ip, CW, IH))
        if sp:
            scans.append(load_fit(sp, CW, SH))
        else:
            box = Image.new("RGB", (CW, SH), (245, 245, 245))
            d = ImageDraw.Draw(box)
            y = SH // 2 - 40
            for i, ln in enumerate(ph.split("\n")):
                f = font(30 if i == 0 else 24)
                tw = d.textlength(ln, font=f)
                d.text(((CW - tw) / 2, y + i * 40), ln, fill=(150, 0, 0) if i == 0 else (90, 90, 90), font=f)
            scans.append(box)

    HDR = 46
    row1 = Image.new("RGB", (CW * 3, HDR + IH), (255, 255, 255))
    row2 = Image.new("RGB", (CW * 3, HDR + SH), (255, 255, 255))
    d1, d2 = ImageDraw.Draw(row1), ImageDraw.Draw(row2)
    for i, ((title, _, _, _), im, sc) in enumerate(zip(cols, imgs, scans)):
        row1.paste(im, (i * CW, HDR))
        d1.text((i * CW + 10, 8), title, fill=(0, 0, 0), font=font(26, True))
        row2.paste(sc, (i * CW, HDR))
        lab = "WinRHIZO 扫描结果" if i > 0 else "扫描结果"
        d2.text((i * CW + 10, 8), lab, fill=(0, 0, 0), font=font(26, True))
        d1.line([(i * CW, 0), (i * CW, HDR + IH)], fill=(200, 200, 200), width=2)
        d2.line([(i * CW, 0), (i * CW, HDR + SH)], fill=(200, 200, 200), width=2)

    # 数据表
    TH = 60 + (len(DATA) + 1) * 46 + 40
    tab = Image.new("RGB", (CW * 3, TH), (255, 255, 255))
    dt = ImageDraw.Draw(tab)
    dt.text((10, 6), "WinRHIZO 扫描数据对比（RHIZO 2016a）", fill=(0, 0, 0), font=font(28, True))
    colx = [20, CW + 20, CW * 2 + 20, CW * 2 + 470]
    heads = ["指标", "② 人工修图", "③ 机器修图", "差异"]
    y = 52
    for x, h in zip(colx, heads):
        dt.text((x, y), h, fill=(0, 0, 0), font=font(24, True))
    dt.line([(0, y + 34), (CW * 3, y + 34)], fill=(120, 120, 120), width=2)
    for i, (k, a, b, diff) in enumerate(DATA):
        yy = y + 44 + i * 46
        if i % 2 == 1:
            dt.rectangle([0, yy - 6, CW * 3, yy + 38], fill=(246, 246, 250))
        dt.text((colx[0], yy), k, fill=(0, 0, 0), font=font(23))
        dt.text((colx[1], yy), a, fill=(0, 0, 0), font=font(23))
        dt.text((colx[2], yy), b, fill=(0, 0, 0), font=font(23))
        col = (200, 0, 0) if diff.startswith("+") else (0, 90, 200)
        dt.text((colx[3], yy), diff, fill=col, font=font(23, True))

    note = ("说明：原图背景不干净，WinRHIZO 无法直接分析，故只能对比「人工修图」与「机器修图」。"
            "两者均为 600 dpi、同一张原图、同一台扫描仪、同一套阈值(灰阶自动)。")
    canvas = Image.new("RGB", (CW * 3, HDR + IH + 16 + HDR + SH + TH + 60), (255, 255, 255))
    yy = 0
    canvas.paste(row1, (0, yy)); yy += HDR + IH + 16
    canvas.paste(row2, (0, yy)); yy += HDR + SH + 16
    canvas.paste(tab, (0, yy)); yy += TH
    dc = ImageDraw.Draw(canvas)
    dc.text((14, yy + 12), note, fill=(90, 90, 90), font=font(22))
    o = OUT / "7-003_扫描对比总图.png"
    canvas.save(o, optimize=True)
    print(f"-> {o}  {canvas.size}")
    return 0


if __name__ == "__main__":
    main()
