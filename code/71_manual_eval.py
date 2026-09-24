"""接入人工真值（man），做四方对比 + 真正的精度评估。

口径（用户 2026-09-24 定义）:
    ORI  = 未经处理、直接扫描的数据
    man  = 人工修图数据（真值）

做的事:
    1. 把 `原图\\` 里的文件按 `-ORI` 后缀对齐命名（与 `-man` 成对）
    2. 对每个**已有人工图**的样本:
         - 四方拼图（ORI | man | CV | U-Net），全分辨率 600dpi，出 TIFF + JPG
         - 以 man 为真值，算 CV / U-Net 的 IoU、Dice、精确率、召回率
    3. 汇总表 + 缩略总览

输出:
    outputs\\对比十张\\四方对比\\全分辨率_TIFF\\*.tif
    outputs\\对比十张\\四方对比\\全分辨率_JPG\\*.jpg
    outputs\\对比十张\\四方对比\\_人工真值评估.csv
    outputs\\对比十张\\四方对比\\_缩略总览.png
"""
from __future__ import annotations

import csv
import statistics as st
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
ORI_DIR = BASE / "原图"
MAN_DIR = BASE / "人工"
OUT = BASE / "四方对比"
OUT_T = OUT / "全分辨率_TIFF"
OUT_J = OUT / "全分辨率_JPG"
GAP = 30
TITLE_H = 130
HEAD_H = 96
NONWHITE = 250          # 非白判根阈值
TARGET_DPI = 600.0
JPEG_Q = 95
PREVIEW_W = 880


def font(sz: int) -> ImageFont.FreeTypeFont:
    for n in ("msyh.ttc", "simhei.ttf", "simsun.ttc"):
        try:
            return ImageFont.truetype(n, sz)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


def rename_ori() -> list[str]:
    """原图\\<n>_<样本>.jpg -> 原图\\<n>_<样本>-ORI.jpg（幂等）"""
    log: list[str] = []
    if not ORI_DIR.exists():
        return log
    for p in sorted(ORI_DIR.glob("*.jpg")):
        if p.stem.endswith("-ORI"):
            continue
        dst = p.with_name(p.stem + "-ORI.jpg")
        if dst.exists():
            continue
        p.rename(dst)
        log.append(f"  {p.name}  ->  {dst.name}")
    return log


def load600(p: Path) -> Image.Image:
    im = Image.open(p).convert("RGB")
    d = im.info.get("dpi")
    dpi = float(d[0]) if d else TARGET_DPI
    if abs(dpi - TARGET_DPI) > 1:
        k = TARGET_DPI / dpi
        im = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), Image.LANCZOS)
    return im


def mask_of(p: Path, thr: int = NONWHITE) -> np.ndarray:
    """判根掩膜。先归一到 600 dpi 网格再阈值——否则 300dpi 的源图(2481×3509)
    和 600dpi 的机器图(4962×7018)尺寸对不上，无法逐像素比。"""
    return np.asarray(load600(p).convert("L")) < thr


def prf(gt: np.ndarray, pr: np.ndarray) -> dict:
    tp = int(np.logical_and(gt, pr).sum())
    fp = int(np.logical_and(~gt, pr).sum())
    fn = int(np.logical_and(gt, ~pr).sum())
    n = int(gt.size)
    return {
        "IoU": tp / (tp + fp + fn) if (tp + fp + fn) else 1.0,
        "Dice": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0,
        "精确率": tp / (tp + fp) if (tp + fp) else 1.0,
        "召回率": tp / (tp + fn) if (tp + fn) else 1.0,
        "TP": tp, "FP": fp, "FN": fn,
        "真值像素": int(gt.sum()), "预测像素": int(pr.sum()), "总像素": n,
    }


def compose(panels: list[Image.Image], title: str, sub: str, heads: list) -> Image.Image:
    w = max(p.width for p in panels)
    h = max(p.height for p in panels)
    total_w = w * len(panels) + GAP * (len(panels) - 1)
    out = Image.new("RGB", (total_w, TITLE_H + HEAD_H + h), (255, 255, 255))
    dr = ImageDraw.Draw(out)
    dr.rectangle([0, 0, total_w, TITLE_H - 1], fill=(232, 232, 232))
    dr.line([0, TITLE_H - 1, total_w, TITLE_H - 1], fill=(140, 140, 140))
    dr.text((24, 30), f"{title}      [{sub}]      {w}×{h} @600dpi", fill=(0, 0, 0), font=font(46))
    dr.rectangle([0, TITLE_H, total_w, TITLE_H + HEAD_H - 1], fill=(248, 248, 248))
    dr.line([0, TITLE_H + HEAD_H - 1, total_w, TITLE_H + HEAD_H - 1], fill=(140, 140, 140))
    for i, (lab, col, extra) in enumerate(heads):
        x = i * (w + GAP)
        txt = f"{lab}    {extra}" if extra else lab
        tw = dr.textlength(txt, font=font(38))
        dr.text((x + max(12, (w - tw) / 2), TITLE_H + 26), txt, fill=col, font=font(38))
    for i, im in enumerate(panels):
        x = i * (w + GAP)
        if im.size != (w, h):
            c = Image.new("RGB", (w, h), (255, 255, 255))
            c.paste(im, (0, 0))
            im = c
        out.paste(im, (x, TITLE_H + HEAD_H))
        dr.rectangle([x, TITLE_H + HEAD_H - 1, x + w - 1, TITLE_H + HEAD_H + h - 1],
                     outline=(150, 150, 150))
    return out


def main() -> int:
    print("[1] 对齐 ORI 命名")
    log = rename_ori()
    print("\n".join(log) if log else "  （无变化，已经是 -ORI 后缀）")

    rows = list(csv.DictReader((BASE / "_抽样清单.csv").open(encoding="utf-8-sig")))
    OUT_T.mkdir(parents=True, exist_ok=True)
    OUT_J.mkdir(parents=True, exist_ok=True)

    have = []
    for r in rows:
        ori = ORI_DIR / f"{r['序号']}_{r['样本']}-ORI.jpg"
        man = MAN_DIR / f"{r['序号']}_{r['样本']}-man.jpg"
        if man.exists():
            have.append((r, ori, man))
    print(f"\n[2] 已有人工图的样本: {len(have)} / {len(rows)}")
    missing = [f"{r['序号']}_{r['样本']}" for r in rows
               if not (MAN_DIR / f"{r['序号']}_{r['样本']}-man.jpg").exists()]
    if missing:
        print(f"    还缺 {len(missing)} 张: {', '.join(missing)}")
    if not have:
        return 1

    # ---- 评估 ----
    print("\n[3] 以 man 为真值评估\n")
    hdr = (f"  {'#':<3}{'样本':<16}{'man根%':>9}{'CV根%':>9}{'UNet根%':>10}"
           f"{'CV_IoU':>9}{'UN_IoU':>9}{'CV_Dice':>9}{'UN_Dice':>9}{'CV召':>7}{'UN召':>7}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    results, csv_rows = [], []
    for r, ori_p, man_p in have:
        idx, stem = r["序号"], r["样本"]
        gt = mask_of(man_p)
        cv = mask_of(Path(r["CV"]))
        un = mask_of(Path(r["UNet"]))
        s_cv, s_un = prf(gt, cv), prf(gt, un)
        n = s_cv["总像素"]
        print(f"  {idx:<3}{stem:<16}{s_cv['真值像素']/n*100:>9.3f}{s_cv['预测像素']/n*100:>9.3f}"
              f"{s_un['预测像素']/n*100:>10.3f}{s_cv['IoU']:>9.4f}{s_un['IoU']:>9.4f}"
              f"{s_cv['Dice']:>9.4f}{s_un['Dice']:>9.4f}"
              f"{s_cv['召回率']:>7.3f}{s_un['召回率']:>7.3f}")
        results.append((r, ori_p, man_p, s_cv, s_un))
        csv_rows.append({
            "序号": idx, "样本": stem, "子目录": r["子目录"],
            "man根占比%": round(s_cv["真值像素"] / n * 100, 4),
            "CV根占比%": round(s_cv["预测像素"] / n * 100, 4),
            "UNet根占比%": round(s_un["预测像素"] / n * 100, 4),
            "CV_IoU": round(s_cv["IoU"], 4), "UNet_IoU": round(s_un["IoU"], 4),
            "CV_Dice": round(s_cv["Dice"], 4), "UNet_Dice": round(s_un["Dice"], 4),
            "CV_精确率": round(s_cv["精确率"], 4), "UNet_精确率": round(s_un["精确率"], 4),
            "CV_召回率": round(s_cv["召回率"], 4), "UNet_召回率": round(s_un["召回率"], 4),
            "CV_FP": s_cv["FP"], "CV_FN": s_cv["FN"],
            "UNet_FP": s_un["FP"], "UNet_FN": s_un["FN"],
            "原图": str(ori_p), "人工": str(man_p), "CV": r["CV"], "UNet": r["UNet"],
        })

    # ---- 汇总 ----
    print("\n[4] 汇总（中位数）")
    print(f"  {'指标':<8}{'CV':>10}{'U-Net':>10}    U-Net 更好的样本数")
    for k in ("IoU", "Dice", "精确率", "召回率"):
        a = [m[3][k] for m in results]
        b = [m[4][k] for m in results]
        win = sum(1 for x, y in zip(a, b) if y > x)
        print(f"  {k:<8}{st.median(a):>10.4f}{st.median(b):>10.4f}    {win}/{len(a)}")

    # ---- 拼图 ----
    print("\n[5] 四方对比图（全分辨率）")
    thumbs = []
    for r, ori_p, man_p, s_cv, s_un in results:
        idx, stem = r["序号"], r["样本"]
        ims = [load600(p) for p in (ori_p, man_p, Path(r["CV"]), Path(r["UNet"]))]
        heads = [("ORI（原始扫描）", (0, 0, 0), ""),
                 ("man（人工修图·真值）", (0, 120, 0), ""),
                 ("CV（经典管线）", (170, 0, 0), f"IoU {s_cv['IoU']:.3f}"),
                 ("U-Net", (0, 60, 150), f"IoU {s_un['IoU']:.3f}")]
        out = compose(ims, f"{idx}   {stem}", r["子目录"], heads)
        tif = OUT_T / f"{idx}_{stem}_ORI-man-CV-UNet.tif"
        jpg = OUT_J / f"{idx}_{stem}_ORI-man-CV-UNet.jpg"
        out.save(tif, format="TIFF", compression="raw", dpi=(600, 600))
        out.save(jpg, format="JPEG", quality=JPEG_Q, subsampling=0, dpi=(600, 600))
        print(f"  {idx}  {stem:<16} {out.width}x{out.height}   "
              f"TIFF {tif.stat().st_size/2**20:6.1f}MB  JPG {jpg.stat().st_size/2**20:5.1f}MB")
        thumbs.append(out.resize((PREVIEW_W, round(out.height * PREVIEW_W / out.width)), Image.LANCZOS))
        del ims

    # 缩略总览
    tw = PREVIEW_W * 4 + GAP * 3
    th = sum(t.height + 8 for t in thumbs)
    sheet = Image.new("RGB", (tw, th), (255, 255, 255))
    y = 0
    for t in thumbs:
        sheet.paste(t, (0, y)); y += t.height + 8
    sheet.save(OUT / "_缩略总览.png")
    print(f"\n缩略总览 -> {OUT / '_缩略总览.png'}   {sheet.width}x{sheet.height}")

    with (OUT / "_人工真值评估.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
        wr.writeheader(); wr.writerows(csv_rows)
    print(f"评估表   -> {OUT / '_人工真值评估.csv'}")
    print(f"TIFF -> {OUT_T}\nJPG  -> {OUT_J}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
