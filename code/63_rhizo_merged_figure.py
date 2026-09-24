"""三样本合并成一张图：行 = 人工 / CV / U-Net，列 = 7-003 / G218-001 / 281-003，底部放数据表。

产出: outputs/Rhizo三方对比/三方合并对比.png
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
import paths as P  # noqa: E402
import tiffsave  # noqa: F401, E402   (确保依赖可用)

Image.MAX_IMAGE_PIXELS = None
OUT = P.PROJ / "outputs" / "Rhizo三方对比"
OUT.mkdir(parents=True, exist_ok=True)
SRC_TXT = Path.home() / "Desktop" / "测试" / "测试数据.TXT"
MM_PER_PX = 25.4 / 600
NONWHITE = 250

SAMPLES = ["7-003", "G218-001", "281-003"]
VERSIONS = ["人工", "CV", "U-Net"]
CLASSES = [0.5, 1.0, 1.5, 2.0, 10.0]
COLORS = [(230, 60, 80), (240, 130, 60), (235, 200, 60), (140, 200, 70), (40, 150, 60)]
VCOLOR = {"人工": (192, 57, 43), "CV": (44, 95, 168), "U-Net": (30, 132, 73)}
IDX = {"根总长 Length (cm)": 15, "分析区面积 (cm²)": 5, "投影面积 ProjArea (cm²)": 17,
       "表面积 SurfArea (cm²)": 19, "平均直径 AvgDiam (mm)": 21,
       "根体积 Volume (cm³)": 25, "根尖数 Tips": 28, "分叉数 Forks": 29, "交叉数 Crossings": 30}


def font(sz, bold=False):
    names = (r"C:\Windows\Fonts\msyhbd.ttc", r"C:\Windows\Fonts\msyh.ttc") if bold \
        else (r"C:\Windows\Fonts\msyh.ttc",)
    for n in names:
        try:
            return ImageFont.truetype(n, sz)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


def read_rows(path: Path):
    raw = path.read_bytes()
    text = None
    for enc in ("gbk", "utf-8-sig", "utf-8", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    lines = [ln for ln in text.splitlines() if ln.strip()]
    hi = next(i for i, ln in enumerate(lines) if ln.startswith("RHIZO"))
    return [ln.split("\t") for ln in lines[hi + 1:]
            if ln.split("\t")[0].strip() and not ln.startswith("SampleId")]


def classify(img: str, sid: str):
    n = img.lower()
    if "unet" in n or "unet" in sid.lower():
        ver = "U-Net"
    elif "-man" in n:
        ver = "人工"
    elif "-mac" in n or "_a_" in n:
        ver = "CV"
    else:
        return None, None
    m = re.match(r"^([0-9A-Za-z\-]+?)[-_](?:man|mac|unet|MAN|MAC|UNet)", img)
    return (m.group(1) if m else None), ver


def find_file(stem: str, ver: str):
    pack = P.PROJ / "outputs" / "扫描对比包"
    cand = P.PROJ / "outputs" / "预标注候选"
    if ver == "人工":
        opts = [pack / f"{stem}_人工.tif", P.TEST_MAN / f"{stem}-man.tif",
                P.TEST_MAN_99GAI / f"{stem}-man-9.22.tif",
                P.PROJ / "人工修复mac图" / f"{stem}-unet-man.tif"]
    elif ver == "CV":
        opts = [pack / f"{stem}_CV.tif", P.TEST_MAC / f"{stem}-mac.tif"]
    else:
        opts = [pack / f"{stem}_UNet.tif", cand / f"{stem}-unet.tif"]
    return next((p for p in opts if p.exists()), None)


def diameter_map(path: Path, w: int) -> Image.Image:
    im = Image.open(path).convert("RGB")
    k = w / im.width
    small = im.resize((w, max(1, int(im.height * k))), Image.LANCZOS)
    a = np.asarray(small)
    mask = np.any(a < NONWHITE, axis=2)
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    width_mm = dt * 2 * MM_PER_PX / k
    out = np.full((*mask.shape, 3), 255, np.uint8)
    for lo, hi, col in zip([0] + CLASSES[:-1], CLASSES, COLORS):
        out[mask & (width_mm > lo) & (width_mm <= hi)] = col
    return Image.fromarray(out)


def main() -> int:
    data = {}
    for c in read_rows(SRC_TXT):
        sid, img = c[0].strip(), (c[3].strip() if len(c) > 3 else "")
        stem, ver = classify(img, sid)
        if stem is None:
            continue
        m = re.match(r"^([0-9A-Za-z\-]+?)[-_](?:man|mac|unet|MAN|MAC|UNet)", img)
        s2 = m.group(1) if m else stem
        if s2 in SAMPLES:
            stem = s2
        data.setdefault(stem, {})[ver] = {k: (c[i] if i < len(c) else "") for k, i in IDX.items()}

    def num(s):
        try:
            return float(s)
        except Exception:  # noqa: BLE001
            return None

    TW = 620          # 每格宽
    IH = 440          # 图高（含标题）
    PAD = 14
    HDR = 34

    grid = Image.new("RGB", (TW * 3 + PAD * 4, HDR + (IH + HDR) * 3 + PAD), (255, 255, 255))
    dr = ImageDraw.Draw(grid)

    # 列标题
    for j, s in enumerate(SAMPLES):
        x = PAD + j * (TW + PAD)
        dr.text((x + TW // 2 - 40, 6), s, fill=(0, 0, 0), font=font(24, True))

    for i, ver in enumerate(VERSIONS):
        y0 = HDR + i * (IH + HDR)
        dr.rectangle([0, y0, TW * 3 + PAD * 4, y0 + HDR], fill=(240, 243, 247))
        dr.text((PAD, y0 + 5), ver, fill=VCOLOR[ver], font=font(24, True))
        for j, s in enumerate(SAMPLES):
            x = PAD + j * (TW + PAD)
            f = find_file(s, ver)
            if f is None:
                im = Image.new("RGB", (TW, IH), (248, 248, 248))
                ImageDraw.Draw(im).text((TW // 2, IH // 2), "缺", fill=(180, 0, 0), font=font(26))
            else:
                dm = diameter_map(f, TW)
                im = Image.new("RGB", (TW, IH), (255, 255, 255))
                im.paste(dm, ((TW - dm.width) // 2, max(0, (IH - dm.height) // 2)))
            grid.paste(im, (x, y0 + HDR))
            dr.rectangle([x, y0 + HDR, x + TW, y0 + HDR + IH], outline=(220, 220, 220), width=1)

    # 色标
    sy = HDR + 3 * (IH + HDR) - 6
    dr.text((PAD, sy - 22), "直径分级(mm)：", fill=(70, 70, 70), font=font(18))
    cx = PAD + 176
    for lo, hi, col in zip([0] + CLASSES[:-1], CLASSES, COLORS):
        dr.rectangle([cx, sy - 20, cx + 44, sy - 4], fill=col)
        lab = f"{hi:g}" if hi < 9 else ">2.0"
        dr.text((cx + 50, sy - 22), lab, fill=(90, 90, 90), font=font(16))
        cx += 150

    # ---- 底部数据表 ----
    ROW_H = 42
    n_rows = len(SAMPLES) * 3
    tab_h = 46 + (n_rows + 1) * ROW_H + 56
    tab = Image.new("RGB", (grid.width, tab_h), (255, 255, 255))
    dt = ImageDraw.Draw(tab)
    dt.text((PAD, 8), "WinRHIZO 扫描数据对比（RHIZO 2016a）", fill=(0, 0, 0), font=font(24, True))
    mkeys = list(IDX.keys())
    cw = [96, 150] + [150] * len(mkeys) + [120]
    heads = ["样本", "版本"] + [k.split()[0] for k in mkeys] + ["根总长 vs 人工"]
    xs, acc = [], PAD
    for w in cw:
        xs.append(acc); acc += w
    ty = 44
    for x, h in zip(xs, heads):
        dt.text((x, ty), h, fill=(0, 0, 0), font=font(18, True))
    dt.line([(0, ty + 28), (tab.width, ty + 28)], fill=(130, 130, 130), width=2)
    r = 0
    for s in SAMPLES:
        base = num(data.get(s, {}).get("人工", {}).get(mkeys[0], ""))
        for ver in VERSIONS:
            yy = ty + 36 + r * ROW_H
            if r % 2 == 1:
                dt.rectangle([0, yy - 6, tab.width, yy + 30], fill=(247, 247, 251))
            if ver == "人工":
                dt.rectangle([0, yy - 6, 6, yy + 30], fill=VCOLOR[ver])
            dt.text((xs[0], yy), s, fill=(0, 0, 0), font=font(18))
            dt.text((xs[1], yy), ver, fill=VCOLOR[ver], font=font(18, True))
            rec = data.get(s, {}).get(ver, {})
            for ci, k in enumerate(mkeys):
                v = num(rec.get(k, ""))
                dt.text((xs[2 + ci], yy), f"{v:.4g}" if v is not None else "—",
                        fill=(0, 0, 0), font=font(18))
            v = num(rec.get(mkeys[0], ""))
            if base and v is not None and ver != "人工":
                pct = (v - base) / base * 100
                col = (200, 0, 0) if abs(pct) >= 10 else ((190, 120, 0) if abs(pct) >= 5
                                                          else (0, 120, 0))
                dt.text((xs[-1], yy), f"{pct:+.1f}%", fill=col, font=font(18, True))
            elif ver == "人工":
                dt.text((xs[-1], yy), "—", fill=(120, 120, 120), font=font(18))
            r += 1
    dt.text((PAD, tab_h - 44),
            "说明：图中颜色为按掩膜局部宽度计算的直径分级（自制可视化，非 WinRHIZO 界面截图）；"
            "偏差 = (该版本 − 人工) / 人工，>10% 标红、5–10% 标橙、<5% 标绿。",
            fill=(110, 110, 110), font=font(16))

    canvas = Image.new("RGB", (grid.width, grid.height + tab.height + 16), (255, 255, 255))
    canvas.paste(grid, (0, 0))
    canvas.paste(tab, (0, grid.height + 16))
    o = OUT / "三方合并对比.png"
    canvas.save(o, optimize=True)
    print(f"-> {o}  {canvas.size}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
