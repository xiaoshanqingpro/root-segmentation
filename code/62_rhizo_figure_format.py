"""按"7-003 扫描对比总图"的格式，做 人工 / CV / U-Net 三方对比图（每个样本一张）。

版式（对照用户给的样例）:
  第 1 排  原图 | 人工 | CV | U-Net
  第 2 排  扫描结果  —— 原图格写"无法分析"；其余三格为**按掩膜宽度着色的直径分级图**
           （自制可视化，非 WinRHIZO 界面截图，图上会注明）
  第 3 排  WinRHIZO 数据表：指标 | 人工 | CV | U-Net | CV偏差 | U-Net偏差

产出: outputs/Rhizo三方对比/<样本>_三方对比.png
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
import paths as P  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
OUT = P.PROJ / "outputs" / "Rhizo三方对比"
OUT.mkdir(parents=True, exist_ok=True)
SRC_TXT = Path.home() / "Desktop" / "测试" / "测试数据.TXT"
MM_PER_PX = 25.4 / 600
NONWHITE = 250

SAMPLES = ["7-003", "G218-001", "281-003"]
VERSIONS = ["人工", "CV", "U-Net"]
# 直径分级边界(mm) 与配色 —— 细 = 红/粉，粗 = 绿（与 WinRHIZO 观感一致）
CLASSES = [0.5, 1.0, 1.5, 2.0, 10.0]
COLORS = [(230, 60, 80), (240, 130, 60), (235, 200, 60), (140, 200, 70), (40, 150, 60)]

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
    import re
    m = re.match(r"^([0-9A-Za-z\-]+?)[-_](?:man|mac|unet|MAN|MAC|UNet)", img)
    return (m.group(1) if m else None), ver


def find_file(stem: str, ver: str) -> Path | None:
    """找对应该样本与版本的图文件"""
    cand = P.PROJ / "outputs" / "预标注候选"
    pack = P.PROJ / "outputs" / "扫描对比包"
    if ver == "人工":
        for p in [pack / f"{stem}_人工.tif", P.TEST_MAN / f"{stem}-man.tif",
                  P.TEST_MAN_99GAI / f"{stem}-man-9.22.tif", P.TEST_MAN_99GAI / f"{stem}-man.tif",
                  P.PROJ / "人工修复mac图" / f"{stem}-unet-man.tif"]:
            if p.exists():
                return p
    elif ver == "CV":
        for p in [pack / f"{stem}_CV.tif", P.TEST_MAC / f"{stem}-mac.tif"]:
            if p.exists():
                return p
    else:
        for p in [pack / f"{stem}_UNet.tif", cand / f"{stem}-unet.tif"]:
            if p.exists():
                return p
    return None


def find_orig(stem: str) -> Path | None:
    for d in [P.TEST_ORIG, P.RAW_D2_99, P.RAW_D2, P.RAW_D1]:
        if not d.exists():
            continue
        for p in d.rglob("*"):
            if p.suffix.lower() in (".tif", ".tiff", ".jpg", ".jpeg") and p.stem == stem:
                return p
    return None


def diameter_map(path: Path, w: int = 660) -> Image.Image:
    """按掩膜局部宽度着色（直径分级）。非 WinRHIZO 截图，是自制可视化。"""
    im = Image.open(path).convert("RGB")
    k = w / im.width
    small = im.resize((w, max(1, int(im.height * k))), Image.LANCZOS)
    a = np.asarray(small)
    mask = np.any(a < NONWHITE, axis=2)
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    width_mm = dt * 2 * MM_PER_PX / k        # 缩略图尺度 -> 原图 mm
    out = np.full((*mask.shape, 3), 255, np.uint8)
    for lo, hi, col in zip([0] + CLASSES[:-1], CLASSES, COLORS):
        sel = mask & (width_mm > lo) & (width_mm <= hi)
        out[sel] = col
    return Image.fromarray(out)


def main() -> int:
    data = {}
    for c in read_rows(SRC_TXT):
        sid, img = c[0].strip(), (c[3].strip() if len(c) > 3 else "")
        stem, ver = classify(img, sid)
        if stem is None:
            continue
        # 用 ImageFileName 找原文件更可靠：记录里 7-003_UNet.tif 的 SampleId 被标成了 218-003-unet
        import re
        m = re.match(r"^([0-9A-Za-z\-]+?)[-_](?:man|mac|unet|MAN|MAC|UNet)", img)
        stem2 = m.group(1) if m else stem
        if stem2 in SAMPLES:
            stem = stem2
        data.setdefault(stem, {})[ver] = {k: (c[i] if i < len(c) else "") for k, i in IDX.items()}

    def num(s):
        try:
            return float(s)
        except Exception:  # noqa: BLE001
            return None

    for stem in SAMPLES:
        print(f"\n=== {stem} ===")
        figs = {}
        for ver in VERSIONS:
            f = find_file(stem, ver)
            figs[ver] = f
            print(f"  {ver:<6} {f.name if f else '缺'}")
        o = find_orig(stem)
        print(f"  原图   {o.name if o else '缺'}")

        # 缩略图
        TW, IH, SH = 640, 470, 430
        def tile_img(p, title, h):
            if p is None:
                im = Image.new("RGB", (TW, h), (245, 245, 245))
                d = ImageDraw.Draw(im)
                d.text((TW // 2 - 40, h // 2), "缺", fill=(180, 0, 0), font=font(28))
            else:
                im = Image.open(p).convert("RGB")
                k = min(TW / im.width, h / im.height)
                im = im.resize((int(im.width * k), int(im.height * k)), Image.LANCZOS)
                bg = Image.new("RGB", (TW, h), (255, 255, 255))
                bg.paste(im, ((TW - im.width) // 2, (h - im.height) // 2))
                im = bg
            c = Image.new("RGB", (TW, h + 34), (255, 255, 255))
            c.paste(im, (0, 34))
            ImageDraw.Draw(c).text((6, 6), title, fill=(0, 0, 0), font=font(20, True))
            return c

        def tile_diam(p, title):
            if p is None:
                return tile_img(None, title, SH)
            im = diameter_map(p, TW)
            bg = Image.new("RGB", (TW, SH), (255, 255, 255))
            bg.paste(im, (0, 0))
            c = Image.new("RGB", (TW, SH + 34), (255, 255, 255))
            c.paste(bg, (0, 34))
            d = ImageDraw.Draw(c)
            d.text((6, 6), title, fill=(0, 0, 0), font=font(20, True))
            # 色标
            x = 8
            for lo, hi, col in zip([0] + CLASSES[:-1], CLASSES, COLORS):
                d.rectangle([x, SH + 12, x + 46, SH + 26], fill=col)
                lab = f"{hi:g}" if hi < 9 else ">2.0"
                d.text((x + 52, SH + 10), lab, fill=(80, 80, 80), font=font(13))
                x += 155
            return c

        row1 = [tile_img(o, "① 原图", IH)] + [tile_img(figs[v], f"{'②③④'[i]} {v}", IH)
                                              for i, v in enumerate(VERSIONS)]
        row2 = []
        # 第一格："原图无法扫描"占位（不要先画"缺"，否则文字重叠）
        blank = Image.new("RGB", (TW, SH + 34), (248, 248, 248))
        dd0 = ImageDraw.Draw(blank)
        dd0.text((6, 6), "扫描结果", fill=(0, 0, 0), font=font(20, True))
        t1 = "原图无法扫描"
        dd0.text((TW // 2 - dd0.textlength(t1, font=font(24, True)) / 2, SH // 2 - 26),
                 t1, fill=(180, 0, 0), font=font(24, True))
        t2 = "（背景不干净，WinRHIZO 无法分析）"
        dd0.text((TW // 2 - dd0.textlength(t2, font=font(16)) / 2, SH // 2 + 12),
                 t2, fill=(120, 120, 120), font=font(16))
        row2.append(blank)
        for v in VERSIONS:
            row2.append(tile_diam(figs[v], f"{v}：直径分级"))

        COLS = 4
        W = TW * COLS + 24
        H1, H2 = row1[0].height, row2[0].height
        TABLE_H = 60 + (len(IDX) + 1) * 44 + 74
        canvas = Image.new("RGB", (W, H1 + H2 + TABLE_H + 30), (255, 255, 255))
        x = 12
        for t in row1:
            canvas.paste(t, (x, 10)); x += TW
        x = 12
        y2 = 10 + H1 + 12
        for t in row2:
            canvas.paste(t, (x, y2)); x += TW

        dr = ImageDraw.Draw(canvas)
        ty = y2 + H2 + 24
        dr.text((14, ty - 26), "WinRHIZO 扫描数据对比（RHIZO 2016a）", fill=(0, 0, 0), font=font(24, True))
        heads = ["指标", "人工", "CV", "U-Net", "CV 偏差", "U-Net 偏差"]
        colx = [14, TW + 14, TW * 2 + 14, TW * 3 + 14, TW * 3 + 250, TW * 3 + 420]
        for x, h in zip(colx, heads):
            dr.text((x, ty), h, fill=(0, 0, 0), font=font(20, True))
        dr.line([(0, ty + 30), (W, ty + 30)], fill=(130, 130, 130), width=2)
        for i, (k, _) in enumerate(IDX.items()):
            yy = ty + 40 + i * 44
            if i % 2 == 1:
                dr.rectangle([0, yy - 6, W, yy + 34], fill=(247, 247, 251))
            base = num(data.get(stem, {}).get("人工", {}).get(k, ""))
            dr.text((colx[0], yy), k, fill=(0, 0, 0), font=font(19))
            for ci, v in enumerate(VERSIONS):
                val = num(data.get(stem, {}).get(v, {}).get(k, ""))
                txt = f"{val:.4g}" if val is not None else "—"
                dr.text((colx[1 + ci], yy), txt, fill=(0, 0, 0), font=font(19))
            for ci, v in enumerate(("CV", "U-Net")):
                val = num(data.get(stem, {}).get(v, {}).get(k, ""))
                if base and val is not None and base != 0:
                    pct = (val - base) / base * 100
                    col = (200, 0, 0) if abs(pct) >= 10 else ((190, 120, 0) if abs(pct) >= 5
                                                              else (0, 120, 0))
                    dr.text((colx[4 + ci], yy), f"{pct:+.1f}%", fill=col, font=font(19, True))
                else:
                    dr.text((colx[4 + ci], yy), "—", fill=(120, 120, 120), font=font(19))
        dr.text((14, ty + 40 + len(IDX) * 44 + 6),
                "注：第二排的「直径分级」由掩膜局部宽度算出，是自制可视化，不是 WinRHIZO 界面截图。"
                "偏差 = (该版本 − 人工) / 人工。",
                fill=(110, 110, 110), font=font(17))
        o2 = OUT / f"{stem}_三方对比.png"
        canvas.save(o2, optimize=True)
        print(f"  -> {o2}  {canvas.size}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
