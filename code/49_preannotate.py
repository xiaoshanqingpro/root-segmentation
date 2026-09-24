"""预标注工具：原图 -> 可交付人工修正的候选白底根图（一条命令）。

**标注流程里的"机器先做一遍"这一步。**

流程:
  1. U-Net 滑窗推理 -> 根概率图 -> 二值掩膜
     （没有权重时自动回退到经典规则管线 rootseg_common）
  2. 边框自动清理（borderclean.py）：扫描仪黑边 + 浅色纸边/盖板边
  3. 细壁闭合环剔除（水渍水泡反光细线）
  4. 非根涂白，输出候选图 + 预览 + 清单

约定:
  - 输出命名 `<样本>-unet.tif`（U-Net 候选），人工修正后另存 `<样本>-man.tif`
  - **不覆盖任何原始文件**
  - 同时输出 `<样本>_cand.png` 预览，便于人工快速过目

用法:
    python 49_preannotate.py --src <原图文件或目录> --out <输出目录> [--limit N]
    python 49_preannotate.py --list        # 只看会处理哪些文件
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
import paths as P  # noqa: E402
import borderclean as BC  # noqa: E402
import tiffsave as TS  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
WEIGHTS = P.PROJ / "outputs" / "训练" / "best_unet.pt"
PATCH, STRIDE = 512, 384
NONWHITE = 250
THR = 0.5
MIN_HOLE_PX = 300
# 候选图格式: False = 未压缩+补齐标签（WinRHIZO 可读）；True = LZW（体积小，仅编辑用）
COMPRESS = False
MAX_WALL_ERODE = 3


def font(sz):
    for n in ("msyh.ttc", "simhei.ttf"):
        try:
            return ImageFont.truetype(n, sz)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


def remove_thin_loops(rgb: np.ndarray) -> tuple[np.ndarray, int]:
    """细壁闭合环（水渍水泡反光细线）剔除"""
    fg = np.any(rgb < NONWHITE, axis=2)
    H, W = fg.shape
    ker = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    holes = ndimage.binary_fill_holes(fg) & ~fg
    if not holes.any():
        return rgb, 0
    nh, lh, sh, _ = cv2.connectedComponentsWithStats(holes.astype(np.uint8), 8)
    big = [(j, sh[j, cv2.CC_STAT_AREA]) for j in range(1, nh)
           if sh[j, cv2.CC_STAT_AREA] >= MIN_HOLE_PX]
    drop = np.zeros_like(fg)
    cnt = 0
    for r in range(1, MAX_WALL_ERODE + 1):
        if not big:
            break
        er = cv2.erode(fg.astype(np.uint8), ker(r)) > 0
        holes_er = ndimage.binary_fill_holes(er) & ~er
        still = []
        for j, area in big:
            m = lh == j
            ys, xs = np.nonzero(m)
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            if holes_er[y0:y1, x0:x1].any():
                still.append((j, area))
            else:
                pad = r + 2
                yy0, yy1 = max(0, y0 - pad), min(H, y1 + pad)
                xx0, xx1 = max(0, x0 - pad), min(W, x1 + pad)
                near = m[yy0:yy1, xx0:xx1]
                drop[yy0:yy1, xx0:xx1] |= fg[yy0:yy1, xx0:xx1] & (
                    cv2.dilate(near.astype(np.uint8), ker(pad)) > 0)
                cnt += 1
        big = still
    if drop.any():
        rgb = rgb.copy()
        rgb[drop] = 255
    return rgb, cnt


def load_unet():
    if not WEIGHTS.exists():
        return None, None
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    import segmentation_models_pytorch as smp
    m = smp.Unet(encoder_name=ck["encoder"], encoder_weights=None,
                 in_channels=3, classes=1).to(dev)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, dev


@torch.no_grad()
def unet_mask(model, dev, img: np.ndarray) -> np.ndarray:
    H, W = img.shape[:2]
    prob = np.zeros((H, W), np.float32)
    cnt = np.zeros((H, W), np.float32)
    ys = list(range(0, max(1, H - PATCH) + 1, STRIDE))
    xs = list(range(0, max(1, W - PATCH) + 1, STRIDE))
    if ys[-1] != H - PATCH:
        ys.append(H - PATCH)
    if xs[-1] != W - PATCH:
        xs.append(W - PATCH)
    tiles, pos = [], []
    for y in ys:
        for x in xs:
            tiles.append(img[y:y + PATCH, x:x + PATCH])
            pos.append((y, x))
    for i in range(0, len(tiles), 8):
        b = np.stack(tiles[i:i + 8]).astype(np.float32) / 255.0
        t = torch.from_numpy(b).permute(0, 3, 1, 2).to(dev)
        with torch.amp.autocast("cuda", enabled=dev.type == "cuda"):
            o = model(t)
        p = torch.sigmoid(o).float().cpu().numpy()[:, 0]
        for k, (y, x) in enumerate(pos[i:i + 8]):
            prob[y:y + PATCH, x:x + PATCH] += p[k]
            cnt[y:y + PATCH, x:x + PATCH] += 1
    return (prob / np.maximum(cnt, 1)) > THR


def collect(src: Path) -> list[Path]:
    if src.is_file():
        return [src]
    exts = (".tif", ".tiff", ".jpg", ".jpeg", ".png")
    fs = [p for p in src.rglob("*") if p.suffix.lower() in exts]
    # 排除已有的 man/mac 产物，只处理"原图"
    return sorted(p for p in fs
                  if "-man" not in p.stem.lower() and "-mac" not in p.stem.lower())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=False)
    ap.add_argument("--out", required=False)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    if not a.src:
        a.src = str(P.RAW_D2)
    src = Path(a.src)
    files = collect(src)
    if a.limit:
        files = files[:a.limit]
    print(f"[预标注] 源 {src}   待处理 {len(files)} 张")
    if a.list:
        for p in files[:40]:
            print(f"  {p}")
        return 0

    out_dir = Path(a.out) if a.out else (P.PROJ / "outputs" / "预标注候选")
    prev_dir = out_dir / "_preview"
    out_dir.mkdir(parents=True, exist_ok=True)
    prev_dir.mkdir(parents=True, exist_ok=True)

    model, dev = load_unet()
    if model is None:
        print("  !! 未找到 U-Net 权重，回退到经典规则管线 rootseg_common")
        import rootseg_common as R
    else:
        print(f"  U-Net 权重 {WEIGHTS.name}   device={dev}")

    rows = []
    for i, p in enumerate(files, 1):
        img = np.asarray(Image.open(p).convert("RGB"))
        if model is not None:
            mask = unet_mask(model, dev, img)
            out = img.copy()
            out[~mask] = 255
        else:
            root, _, _ = R.extract_root(img, return_maps=False)
            out = img.copy()
            out[~root] = 255

        out, bc = BC.clean_borders(out)
        out, n_loops = remove_thin_loops(out)

        dst = out_dir / (p.stem + "-unet.tif")
        TS.save_tiff_winrhizo(out, dst, compress=COMPRESS)
        fg = float(np.any(out < NONWHITE, axis=2).mean() * 100)

        pv = Image.fromarray(out)
        PW = 900
        pv = pv.resize((PW, int(pv.height * PW / pv.width)), Image.LANCZOS)
        c = Image.new("RGB", (PW, pv.height + 54), (255, 255, 255))
        c.paste(pv, (0, 54))
        d = ImageDraw.Draw(c)
        d.text((8, 6), f"机器候选  {p.name}  ->  {dst.name}", fill=(0, 0, 0), font=font(20))
        d.text((8, 30), f"根占比 {fg:.3f}%   清边框 {bc['removed_total_px']:,}px   "
                        f"清细线环 {n_loops} 处", fill=(150, 0, 0), font=font(17))
        c.save(prev_dir / f"{p.stem}_cand.png", optimize=True)

        rows.append({"src": str(p), "candidate": str(dst), "root_pct": round(fg, 3),
                     "removed_border_px": bc["removed_total_px"],
                     "removed_loops": n_loops})
        print(f"  [{i}/{len(files)}] {p.name:<20} 根 {fg:6.3f}%  "
              f"边框剔除 {bc['removed_total_px']:>9,}px  细线环 {n_loops}", flush=True)

    if rows:
        with (out_dir / "candidates.csv").open("w", newline="", encoding="utf-8-sig") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
    print(f"\n  候选 -> {out_dir}")
    print(f"  预览 -> {prev_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
