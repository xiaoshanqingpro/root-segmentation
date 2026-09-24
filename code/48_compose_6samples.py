"""6 张样本总对比图：人工修图 / 机器修图 并排 + 四项重点指标的数值与差距。

产出: outputs/端到端对比/6张总对比.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# --- 统一路径（见 paths.py；数据搬家只改那里）---
sys.path.insert(0, str(Path(__file__).parent))
import paths as P  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

Image.MAX_IMAGE_PIXELS = None
TEST = P.TEST
OUT = Path(r"D:\根系分割项目\outputs\端到端对比")

# 样本: (显示名, 原图, 人工图, 机器图, 人工记录id, 机器记录id, 特征说明)
SAMPLES = [
    ("28-CL-003", "28-CL-003.jpg", "28-CL-003-man.tif", "28-CL-003-mac.tif",
     "28-CL-003-man", "28-CL-003-mac", "根最黑最饱和"),
    ("126-003", "126-003.jpg", "126-003-man.tif", "126-003-mac.tif",
     "126-003-man", "126-003-mac", "根最粗 30px"),
    ("134-003", "134-003.jpg", "134-003-man.tif", "134-003-mac.tif",
     "134-003-man", "134-003-mac", "根系最密 23%"),
    ("66-003", "66-003.jpg", "66-003-man.tif", "66-003-mac.tif",
     "66-003-man", "66-003-mac", "阴影最重 13.6%"),
    ("G218-001", "G218-001.jpg", "G218-001-man.tif", "G218-001-mac.tif",
     "G218-001-man", "G218-001-MAC", "根最淡（灰度163）"),
    ("7-003", "7-003.tif", "7-003-man.tif", "7-003-mac.tif",
     "7003man", "7003auto2-tiff", "旧参数，仅作参考"),
]
KEYS = [("Length(cm)", "根总长", "cm"), ("SurfArea(cm2)", "表面积", "cm²"),
        ("RootVolume(cm3)", "根体积", "cm³"), ("AvgDiam(mm)", "平均直径", "mm")]


def font(sz, b=False):
    for n in (("msyhbd.ttc", "msyh.ttc") if b else ("msyh.ttc",)):
        try:
            return ImageFont.truetype(n, sz)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


def parse():
    txt = (TEST / "测试数据.TXT").read_text(encoding="gbk", errors="replace")
    lines = [l.rstrip("\n") for l in txt.splitlines()]
    names = [h.strip() for h in next(l for l in lines if "SoilVol(m3)" in l).split("\t")]
    recs = {}
    for l in lines:
        f = l.split("\t")
        if len(f) < 5 or f[0].strip() in ("SampleId",):
            continue
        row = {}
        for k, _, _ in KEYS:
            try:
                row[k] = float(f[names.index(k)])
            except Exception:  # noqa: BLE001
                row[k] = float("nan")
        recs[f[0].strip()] = row
    return recs


def thumb(p: Path, w: int, h: int) -> Image.Image:
    """缩略图用**分块取最暗**而不是普通缩放。

    原因: 原图 7019px 缩到 ~460px（系数 0.065），一根 5px 宽的细根只剩 0.3px，
    普通插值会把它平均掉 -> 细根稀疏的图（66-003 / G218-001 / 7-003）缩略图上直接看不见。
    分块取最暗能保住细的深色结构。
    """
    im = Image.open(p).convert("RGB")
    a = np.asarray(im)
    k = max(1, int(np.ceil(max(a.shape[:2]) / max(w, h))))
    H, W = a.shape[:2]
    Hc, Wc = (H // k) * k, (W // k) * k
    a = a[:Hc, :Wc]
    blocks = a.reshape(Hc // k, k, Wc // k, k, 3)
    small = blocks.min(axis=(1, 3))          # 每块取最暗像素 -> 保住细根
    im2 = Image.fromarray(small.astype(np.uint8))
    k2 = min(w / im2.width, h / im2.height)
    if k2 < 1:
        im2 = im2.resize((int(im2.width * k2), int(im2.height * k2)), Image.LANCZOS)
    bg = Image.new("RGB", (w, h), (255, 255, 255))
    bg.paste(im2, ((w - im2.width) // 2, (h - im2.height) // 2))
    return bg


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    recs = parse()
    CW, TH = 470, 330
    PAD = 8
    HDR = 44
    W = CW * 6 + PAD * 2
    # 高: 标题 + 人工行 + 机器行 + 数据表
    TH_TITLE = 70
    ROW_LBL = 36
    ROWS = [("原图（扫描件）", "原图", 1, (238, 238, 238)),
            ("人工修图", "人工", 2, (240, 240, 245)),
            ("机器修图", "机器", 3, (235, 245, 250))]
    TBL_H = 56 + len(SAMPLES) * 104 + 40
    H = TH_TITLE + len(ROWS) * (ROW_LBL + TH + PAD) + TBL_H + 30
    fig = Image.new("RGB", (W, H), (255, 255, 255))
    dr = ImageDraw.Draw(fig)

    dr.text((PAD + 6, 12), "6 张样本端到端对比：原图 / 人工修图 / 机器修图",
            fill=(0, 0, 0), font=font(30, True))
    dr.text((PAD + 6, 46), "同一张原图 / 同一台扫描仪 / 同一套阈值（RHIZO 2016a）"
                           "　　差异 = (机器 − 人工) / 人工", fill=(110, 110, 110), font=font(19))

    y = TH_TITLE
    for tag, folder, idx, bgc in ROWS:
        dr.rectangle([0, y, W, y + ROW_LBL], fill=bgc)
        dr.text((PAD + 6, y + 7), tag, fill=(0, 0, 0), font=font(22, True))
        for i, s in enumerate(SAMPLES):
            x = PAD + i * CW
            dr.text((x + 6, y + 9), f"{i+1}. {s[0]}　{s[6]}", fill=(140, 0, 0), font=font(17))
        y += ROW_LBL
        for i, s in enumerate(SAMPLES):
            src = TEST / folder / s[idx]
            if src.exists():
                fig.paste(thumb(src, CW - PAD * 2, TH), (PAD + i * CW + PAD, y))
            else:
                dr.rectangle([PAD + i * CW + PAD, y, PAD + i * CW + CW - PAD, y + TH],
                             fill=(250, 235, 235))
                dr.text((PAD + i * CW + 20, y + TH // 2), f"缺: {src.name}",
                        fill=(200, 0, 0), font=font(18))
            dr.rectangle([PAD + i * CW, y, PAD + i * CW + CW, y + TH], outline=(215, 215, 215))
        y += TH + PAD

    # ---- 数据表 ----
    y += 12
    dr.text((PAD + 6, y), "四项重点指标对比", fill=(0, 0, 0), font=font(24, True))
    y += 40
    hdr_y = y
    dr.rectangle([PAD, hdr_y, W - PAD, hdr_y + 40], fill=(245, 245, 245))
    dr.text((PAD + 10, hdr_y + 9), "样本", fill=(0, 0, 0), font=font(20, True))
    colw = (W - PAD * 2 - 210) // 4
    for j, (k, cn, unit) in enumerate(KEYS):
        dr.text((PAD + 210 + j * colw + 10, hdr_y + 9), f"{cn}（{unit}）",
                fill=(0, 0, 0), font=font(20, True))
    y = hdr_y + 40

    for i, s in enumerate(SAMPLES):
        name, man_id, mac_id = s[0], s[4], s[5]
        rh = 104
        if i % 2 == 1:
            dr.rectangle([PAD, y, W - PAD, y + rh], fill=(249, 249, 252))
        dr.text((PAD + 10, y + rh // 2 - 12), f"{i+1}. {name}", fill=(0, 0, 0), font=font(21, True))
        m, k = recs.get(man_id, {}), recs.get(mac_id, {})
        for j, (key, cn, unit) in enumerate(KEYS):
            x = PAD + 210 + j * colw
            a, b = m.get(key, float("nan")), k.get(key, float("nan"))
            if np.isnan(a) or np.isnan(b):
                continue
            d = 100 * (b / a - 1)
            dr.text((x + 10, y + 10), f"人工  {a:g}", fill=(70, 70, 70), font=font(20))
            dr.text((x + 10, y + 38), f"机器  {b:g}", fill=(0, 90, 160), font=font(20))
            col = (0, 140, 0) if abs(d) < 10 else ((210, 120, 0) if abs(d) < 20 else (205, 0, 0))
            dr.text((x + 10, y + 66), f"差异  {d:+.1f}%", fill=col, font=font(22, True))
        dr.line([(PAD, y + rh), (W - PAD, y + rh)], fill=(225, 225, 225))
        y += rh

    # 汇总行
    y += 6
    dr.text((PAD + 10, y), "中位差异", fill=(0, 0, 0), font=font(21, True))
    for j, (key, cn, unit) in enumerate(KEYS):
        x = PAD + 210 + j * colw
        ds = []
        for s in SAMPLES:
            a, b = recs.get(s[4], {}).get(key), recs.get(s[5], {}).get(key)
            if a and b:
                ds.append(100 * (b / a - 1))
        if not ds:
            continue
        med = float(np.median(ds))
        col = (0, 140, 0) if abs(med) < 10 else ((210, 120, 0) if abs(med) < 20 else (205, 0, 0))
        dr.text((x + 10, y), f"{med:+.1f}%", fill=col, font=font(24, True))
        dr.text((x + 130, y + 3), f"（{len(ds)} 张）", fill=(150, 150, 150), font=font(16))

    o = OUT / "6张总对比.png"
    fig.save(o, optimize=True)
    print(f"-> {o}  {fig.size}")
    return 0


if __name__ == "__main__":
    main()
