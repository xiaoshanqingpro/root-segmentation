"""随机抽 10 张，生成「原图 | CV | U-Net」三方对比图，供人工目视判断。

输出:
    outputs\\对比十张\\十张_原图_CV_UNet_对比.png     总览大图（10 行 × 3 列）
    outputs\\对比十张\\单张\\<样本>_三方.png           每张单独一张，分辨率更高好放大
    outputs\\对比十张\\_抽样清单.csv                  抽到哪 10 张、各自根占比

口径说明（重要）:
    - CV 产出是**纯黑白二值图**（根=0 黑 / 背景=255 白）→ 对比图里呈黑色剪影
    - U-Net 产出是**保留原色的白底图**（根保留褐色 / 背景涂白）→ 呈褐色
    两者是同一口径的"哪些像素算根"，只是着色方式不同；根占比按同一规则（非白像素）统计。
"""
from __future__ import annotations

import csv
import random
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
PROJ = P.PROJ
CV = PROJ / "机器图_CV"
UN = PROJ / "机器图_UNet"
OUT = PROJ / "outputs" / "对比十张"
DESKTOP = Path.home() / "Desktop"
SHANKUO = DESKTOP / "杉阔混交林"
RAW1 = DESKTOP / "扫描" / "图片"
RAW2 = DESKTOP / "扫描" / "原始图片"
N_SAMPLE = 10
SEED = 20260924
NONWHITE = 250
W_SHEET = 600      # 总览图里每张子图宽
W_SINGLE = 900     # 单张图里每张子图宽
LABEL_H = 40
GAP = 6


# ---------- 建立索引 ----------
def build_index() -> dict[str, dict]:
    idx: dict[str, dict] = {}
    for p in CV.rglob("*.tif"):
        if p.name.endswith("-mac.tif"):
            idx.setdefault(p.name[:-8], {})["cv"] = p
    for p in UN.rglob("*.tif"):
        if p.name.endswith("-mac.tif"):
            idx.setdefault(p.name[:-8], {})["un"] = p
        elif p.name.endswith("-unet.tif"):
            idx.setdefault(p.name[:-9], {})["un"] = p
    return idx


def find_original(stem: str, cv_path: Path | None, un_path: Path | None) -> Path | None:
    """按 CV/UNet 所在的数据集结构反查原图。"""
    for p, roots in ((cv_path, [CV]), (un_path, [UN])):
        if p is None:
            continue
        rel = None
        for r in roots:
            try:
                rel = p.relative_to(r)
                break
            except ValueError:
                continue
        if rel is None:
            continue
        parts = rel.parts
        head = parts[0]
        tails = parts[1:-1]
        if head == "数据集1":
            base = RAW1
        elif head == "数据集2":
            base = RAW2
        elif head == "杉阔混交林":
            base = SHANKUO
        else:
            continue
        d = base.joinpath(*tails)
        if d.is_dir():
            for f in sorted(d.iterdir()):
                if f.is_file() and f.stem == stem and f.suffix.lower() in (
                        ".jpg", ".jpeg", ".tif", ".tiff", ".png"):
                    return f
    # 兜底：全盘按文件名找
    for base in (SHANKUO, RAW1, RAW2):
        for f in base.rglob(stem + ".*"):
            if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".tif", ".tiff", ".png"):
                return f
    return None


def root_pct(p: Path) -> float:
    a = np.asarray(Image.open(p).convert("L"))
    return float((a < NONWHITE).mean() * 100)


def thumb(p: Path, w: int) -> Image.Image:
    im = Image.open(p).convert("RGB")
    h = max(1, round(im.height * w / im.width))
    return im.resize((w, h), Image.LANCZOS)


def font(sz: int) -> ImageFont.FreeTypeFont:
    for n in ("msyh.ttc", "simhei.ttf", "simsun.ttc"):
        try:
            return ImageFont.truetype(n, sz)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


def make_row(name: str, sub: str, imgs: list[Image.Image], pcts: list[str], w: int) -> Image.Image:
    h = max(i.height for i in imgs)
    total_w = w * 3 + GAP * 2
    row = Image.new("RGB", (total_w, LABEL_H + h), (255, 255, 255))
    dr = ImageDraw.Draw(row)
    dr.rectangle([0, 0, total_w, LABEL_H - 1], fill=(235, 235, 235))
    dr.line([0, LABEL_H - 1, total_w, LABEL_H - 1], fill=(160, 160, 160))
    f = font(19)
    dr.text((8, 9), f"{name}    [{sub}]", fill=(0, 0, 0), font=f)
    for i, (im, lab, pc) in enumerate(zip(imgs, ["原图", "CV（经典管线）", "U-Net"], pcts)):
        x = i * (w + GAP)
        row.paste(im, (x, LABEL_H))
        dr.rectangle([x, LABEL_H, x + w - 1, LABEL_H + h - 1], outline=(190, 190, 190))
        f2 = font(17)
        txt = f"{lab}   根占比 {pc}" if pc else lab
        tw = dr.textlength(txt, font=f2)
        dr.rectangle([x + 6, LABEL_H + 6, x + 14 + tw, LABEL_H + 32], fill=(255, 255, 255))
        dr.text((x + 10, LABEL_H + 9), txt, fill=(180, 0, 0), font=f2)
    return row


def main() -> int:
    random.seed(SEED)
    idx = build_index()
    common = sorted(s for s, v in idx.items() if "cv" in v and "un" in v)
    print(f"CV/UNet 两边都有的样本: {len(common)}")

    # 解析原图，剔除找不到的
    ready = []
    for s in common:
        o = find_original(s, idx[s].get("cv"), idx[s].get("un"))
        if o is not None:
            ready.append((s, o, idx[s]["cv"], idx[s]["un"]))
    print(f"其中原图也能找到的: {len(ready)}")

    picked = random.sample(ready, min(N_SAMPLE, len(ready)))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "单张").mkdir(parents=True, exist_ok=True)

    rows_sheet, rows_single = [], []
    meta = []
    for k, (stem, orig, cvp, unp) in enumerate(picked, 1):
        pc_cv = root_pct(cvp)
        pc_un = root_pct(unp)
        try:
            dset = cvp.relative_to(CV).parts[0]
            sub = str(cvp.relative_to(CV).parent)
        except ValueError:
            dset, sub = "?", "?"
        # 原图是哪天/哪批
        try:
            batch = str(orig.parent.relative_to(orig.parents[len(orig.parents) - 1])) if False else orig.parent.name
        except Exception:  # noqa: BLE001
            batch = ""
        label = f"{k:02d}  {stem}"
        imgs_sheet = [thumb(orig, W_SHEET), thumb(cvp, W_SHEET), thumb(unp, W_SHEET)]
        imgs_single = [thumb(orig, W_SINGLE), thumb(cvp, W_SINGLE), thumb(unp, W_SINGLE)]
        pcts = ["", f"{pc_cv:.3f}%", f"{pc_un:.3f}%"]
        rows_sheet.append(make_row(label, sub, imgs_sheet, pcts, W_SHEET))
        rows_single.append((f"{k:02d}_{stem}", sub, imgs_single, pcts))
        meta.append({"序号": k, "样本": stem, "数据集": dset, "子目录": sub, "原图": str(orig),
                     "CV": str(cvp), "UNet": str(unp),
                     "CV根占比%": round(pc_cv, 4), "UNet根占比%": round(pc_un, 4),
                     "倍数UNet/CV": round(pc_un / pc_cv, 2) if pc_cv > 0 else float("inf")})
        print(f"  {k:02d} {stem:<20} {sub:<28} CV {pc_cv:6.3f}%   U-Net {pc_un:6.3f}%")

    # 总览大图
    sw = W_SHEET * 3 + GAP * 2
    sh = sum(r.height + GAP for r in rows_sheet)
    sheet = Image.new("RGB", (sw, sh), (255, 255, 255))
    y = 0
    for r in rows_sheet:
        sheet.paste(r, (0, y)); y += r.height + GAP
    f_main = OUT / "十张_原图_CV_UNet_对比.png"
    sheet.save(f_main)
    print(f"\n总览图 -> {f_main}   {sheet.width}x{sheet.height}")

    # 单张
    for name, sub, imgs, pcts in rows_single:
        row = make_row(name, sub, imgs, pcts, W_SINGLE)
        row.save(OUT / "单张" / f"{name}_三方.png")
    print(f"单张图 -> {OUT / '单张'}   （{len(rows_single)} 张）")

    with (OUT / "_抽样清单.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(meta[0].keys()))
        wr.writeheader(); wr.writerows(meta)
    print(f"清单   -> {OUT / '_抽样清单.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
