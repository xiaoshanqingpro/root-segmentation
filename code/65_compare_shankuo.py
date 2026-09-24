"""杉阔混交林：U-Net 输出 vs 经典管线输出 vs 原图 —— 三方对比。

目的: 本次 U-Net 跑出的根占比中位 0.594%，而 `原始图片` 批次是 2.003%，
      低 3.4 倍。要判断是"这批根本来就少"还是"U-Net 在此数据集上漏检"。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as P  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
SRC = Path.home() / "Desktop" / "杉阔混交林"
MAC_CLS = P.PROJ / "机器图_杉阔混交林"          # 经典管线
MAC_UNET = P.PROJ / "机器图_杉阔混交林_unet"     # U-Net
OUT = P.PROJ / "outputs" / "杉阔混交林对比"
NONWHITE = 250


def font(sz):
    for n in ("msyh.ttc", "simhei.ttf"):
        try:
            return ImageFont.truetype(n, sz)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


def fg_pct(p: Path) -> float:
    a = np.asarray(Image.open(p).convert("RGB"))
    return float(np.any(a < NONWHITE, axis=2).mean() * 100)


def tile(arr, title, w=620):
    im = Image.fromarray(arr)
    im = im.resize((w, max(1, int(im.height * w / im.width))), Image.LANCZOS)
    c = Image.new("RGB", (w, im.height + 34), (255, 255, 255))
    c.paste(im, (0, 34))
    ImageDraw.Draw(c).text((6, 6), title, fill=(0, 0, 0), font=font(18))
    return c


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    # 挑几个样本：9.8校准 一张、F 两张、F补 一张
    picks = ["9.8校准/1001-001", "F/11-CL-002", "F/3004-001", "F补/20-B-QG-001"]
    tiles = []
    for rel in picks:
        o = SRC / (rel + ".jpg")
        if not o.exists():
            cand = list((SRC / rel).parent.glob(Path(rel).name + ".*"))
            if not cand:
                print(f"  跳过 {rel}（找不到原图）")
                continue
            o = cand[0]
        stem = Path(rel).name
        sub = str(Path(rel).parent)
        c = MAC_CLS / sub / f"{stem}-mac.tif"
        u = MAC_UNET / sub / f"{stem}-mac.tif"
        if not c.exists() or not u.exists():
            print(f"  跳过 {rel}（缺输出 c={c.exists()} u={u.exists()}）")
            continue
        orig = np.asarray(Image.open(o).convert("RGB"))
        cimg = np.asarray(Image.open(c).convert("RGB"))
        uimg = np.asarray(Image.open(u).convert("RGB"))
        pc, pu = fg_pct(c), fg_pct(u)
        row = [tile(orig, f"{rel}  原图"),
               tile(cimg, f"经典管线  根 {pc:.3f}%"),
               tile(uimg, f"U-Net  根 {pu:.3f}%")]
        h = max(t.height for t in row)
        r = Image.new("RGB", (sum(t.width for t in row), h), (255, 255, 255))
        x = 0
        for t in row:
            r.paste(t, (x, 0)); x += t.width
        tiles.append(r)
        print(f"  {rel:<24} 经典 {pc:6.3f}%   U-Net {pu:6.3f}%   "
              f"U-Net/经典 = {pu/max(1e-9,pc):.2f}x")

    if tiles:
        W = max(t.width for t in tiles)
        canvas = Image.new("RGB", (W, sum(t.height for t in tiles) + 10 * len(tiles)),
                           (255, 255, 255))
        y = 0
        for t in tiles:
            canvas.paste(t, (0, y)); y += t.height + 10
        o = OUT / "杉阔混交林_三方对比.png"
        canvas.save(o, optimize=True)
        print(f"\n  -> {o}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
