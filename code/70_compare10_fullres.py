"""把 10 张三方对比图重出为**原图分辨率（600 dpi 网格）**，输出 TIFF + JPG。

与 `69_compare10.py` 的区别:
    - 不再缩略，三栏都按 **600 dpi 原始像素**拼（约 4962×7019 或 7019×4962 每栏）
    - 300 dpi 的源图（第 01 张 11-QG-003 是 2481×3509）上采样到 600 dpi 网格，
      与管线喂给模型的口径一致，三栏才能逐像素对齐
    - 标注只放在顶部栏，不在图上压字，保持图像干净

输出:
    outputs\\对比十张\\全分辨率_TIFF\\<序号>_<样本>_原图-CV-UNet.tif   未压缩，无损
    outputs\\对比十张\\全分辨率_JPG\\ <序号>_<样本>_原图-CV-UNet.jpg   q95，方便快速打开
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
import paths as P  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

Image.MAX_IMAGE_PIXELS = None
BASE = P.PROJ / "outputs" / "对比十张"
OUT_TIF = BASE / "全分辨率_TIFF"
OUT_JPG = BASE / "全分辨率_JPG"
GAP = 30           # 栏间距
TITLE_H = 130      # 顶部标题栏
HEAD_H = 96        # 列标题栏
NONWHITE = 250
TARGET_DPI = 600.0
JPEG_Q = 95


def font(sz: int) -> ImageFont.FreeTypeFont:
    for n in ("msyh.ttc", "simhei.ttf", "simsun.ttc"):
        try:
            return ImageFont.truetype(n, sz)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


def load600(p: Path) -> tuple[Image.Image, str]:
    """读图并归一到 600 dpi 像素网格。返回 (图, 说明)"""
    im = Image.open(p).convert("RGB")
    d = im.info.get("dpi")
    dpi = float(d[0]) if d else TARGET_DPI
    note = ""
    if abs(dpi - TARGET_DPI) > 1:
        k = TARGET_DPI / dpi
        new = (max(1, round(im.width * k)), max(1, round(im.height * k)))
        im = im.resize(new, Image.LANCZOS)
        note = f"{dpi:.0f}dpi→600dpi"
    return im, note


def root_pct(im: Image.Image) -> float:
    a = np.asarray(im.convert("L"))
    return float((a < NONWHITE).mean() * 100)


def main() -> int:
    rows = list(csv.DictReader((BASE / "_抽样清单.csv").open(encoding="utf-8-sig")))
    OUT_TIF.mkdir(parents=True, exist_ok=True)
    OUT_JPG.mkdir(parents=True, exist_ok=True)

    f_title, f_head = font(46), font(38)
    manifest = []

    for r in rows:
        idx, stem, sub = r["序号"], r["样本"], r["子目录"]
        p_orig, p_cv, p_un = Path(r["原图"]), Path(r["CV"]), Path(r["UNet"])

        im_o, note_o = load600(p_orig)
        im_c, _ = load600(p_cv)
        im_u, _ = load600(p_un)

        # 三栏尺寸对齐（理论上 600dpi 网格下应完全一致）
        w = max(im_o.width, im_c.width, im_u.width)
        h = max(im_o.height, im_c.height, im_u.height)
        panels = []
        for im in (im_o, im_c, im_u):
            if im.size != (w, h):
                canvas = Image.new("RGB", (w, h), (255, 255, 255))
                canvas.paste(im, (0, 0))
                im = canvas
            panels.append(im)

        pc_cv, pc_un = root_pct(panels[1]), root_pct(panels[2])

        total_w = w * 3 + GAP * 2
        out = Image.new("RGB", (total_w, TITLE_H + HEAD_H + h), (255, 255, 255))
        dr = ImageDraw.Draw(out)

        # 顶部标题栏
        dr.rectangle([0, 0, total_w, TITLE_H - 1], fill=(232, 232, 232))
        dr.line([0, TITLE_H - 1, total_w, TITLE_H - 1], fill=(140, 140, 140))
        dr.text((24, 30), f"{idx}   {stem}      [{sub}]      {w}×{h} @600dpi",
                fill=(0, 0, 0), font=f_title)
        if note_o:
            dr.text((total_w - 480, 44), f"原图 {note_o}", fill=(180, 0, 0), font=font(32))

        # 列标题栏
        dr.rectangle([0, TITLE_H, total_w, TITLE_H + HEAD_H - 1], fill=(248, 248, 248))
        dr.line([0, TITLE_H + HEAD_H - 1, total_w, TITLE_H + HEAD_H - 1], fill=(140, 140, 140))
        heads = [("原图", (0, 0, 0), ""),
                 ("CV（经典管线）", (170, 0, 0), f"根占比 {pc_cv:.3f}%"),
                 ("U-Net", (0, 60, 150), f"根占比 {pc_un:.3f}%")]
        for i, (lab, col, extra) in enumerate(heads):
            x = i * (w + GAP)
            txt = f"{lab}    {extra}" if extra else lab
            tw = dr.textlength(txt, font=f_head)
            dr.text((x + max(12, (w - tw) / 2), TITLE_H + 26), txt, fill=col, font=f_head)

        # 贴图
        for i, im in enumerate(panels):
            out.paste(im, (i * (w + GAP), TITLE_H + HEAD_H))
            dr.rectangle([i * (w + GAP), TITLE_H + HEAD_H - 1,
                          i * (w + GAP) + w - 1, TITLE_H + HEAD_H + h - 1], outline=(150, 150, 150))

        name = f"{idx}_{stem}_原图-CV-UNet"
        tif_p = OUT_TIF / (name + ".tif")
        jpg_p = OUT_JPG / (name + ".jpg")
        out.save(tif_p, format="TIFF", compression="raw", dpi=(600, 600))
        out.save(jpg_p, format="JPEG", quality=JPEG_Q, subsampling=0, dpi=(600, 600))

        mb_t = tif_p.stat().st_size / 1e6
        mb_j = jpg_p.stat().st_size / 1e6
        print(f"  {idx}  {stem:<16} {out.width}x{out.height}   "
              f"CV {pc_cv:6.3f}%  U-Net {pc_un:6.3f}%   TIFF {mb_t:7.1f}MB  JPG {mb_j:6.1f}MB"
              + (f"   [{note_o}]" if note_o else ""))
        manifest.append({"序号": idx, "样本": stem, "子目录": sub, "画布": f"{out.width}x{out.height}",
                         "根占比CV%": round(pc_cv, 4), "根占比UNet%": round(pc_un, 4),
                         "TIFF": str(tif_p), "JPG": str(jpg_p),
                         "TIFF_MB": round(mb_t, 1), "JPG_MB": round(mb_j, 1)})

    with (BASE / "_全分辨率清单.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(manifest[0].keys()))
        wr.writeheader()
        wr.writerows(manifest)

    st = sum(m["TIFF_MB"] for m in manifest)
    sj = sum(m["JPG_MB"] for m in manifest)
    print(f"\nTIFF -> {OUT_TIF}   {len(manifest)} 个，共 {st/1000:.2f} GB")
    print(f"JPG  -> {OUT_JPG}   {len(manifest)} 个，共 {sj/1000:.2f} GB")
    print(f"清单 -> {BASE / '_全分辨率清单.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
