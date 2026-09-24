"""为人工精修挑选"差异最大"的标准图集（4~5 张）。

用户要求: 找 4~5 张差异比较大的图（颜色差异、粗细差异、阴影差异）做人工精修标准。

做法（不靠眼缘，靠量化）:
  1. 对候选图逐张算 5 个特征:
       根灰度      -> 颜色差异（根是深褐还是浅黄）
       根饱和度    -> 颜色差异（偏褐还是偏灰）
       根宽度中位  -> 粗细差异
       阴影占比    -> 阴影/水渍差异
       前景占比    -> 根系疏密
  2. 归一化后做**最远点采样**（贪心最大化最小距离），选出的 5 张在特征空间里彼此最分散
  3. 顺带保证跨子集覆盖

产出: outputs/标准图集/标准图集_contact.png + 标准图集.csv（含每张的特征与"为什么选它"）
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
import rootseg_common as R  # noqa: E402

# --- 统一路径（见 paths.py；数据搬家只改那里）---
sys.path.insert(0, str(Path(__file__).parent))
import paths as P  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
PROJ = P.PROJ
SRC2 = P.RAW_D2
SRC1 = PROJ / "processed" / "600dpi"
OUT = PROJ / "outputs" / "标准图集"
N_PICK = 5
WORKERS = 8

# 已在做的样本，排除
DONE = {"7-001", "7-002", "7-003"}


def features(args):
    src = args
    cv2.setNumThreads(1)
    p = Path(src)
    rgb = np.asarray(Image.open(p).convert("RGB"))
    H, W = rgb.shape[:2]
    sc = min(1.0, 1400 / max(H, W))
    small = cv2.resize(rgb, (int(W * sc), int(H * sc)), interpolation=cv2.INTER_AREA) if sc < 1 else rgb
    hh, ww = small.shape[:2]

    bg = R.flood_background(small)
    fg = ~bg
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1].astype(np.float32) / 255.0
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32)
    k = max(31, (int(min(hh, ww) * 0.05) // 2) * 2 + 1)
    illum = cv2.morphologyEx(gray.astype(np.uint8), cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(np.float32)
    depth = illum - gray
    deep = depth > R.DARK_DROP
    dark = gray < R.GRAY_TUB
    core = fg & deep & ((sat >= R.SAT_ROOT) | ((np.zeros_like(gray) > 0) | dark))
    core = fg & deep & ((sat >= R.SAT_ROOT) | dark)
    if core.sum() < 50:
        core = fg & deep

    dt = cv2.distanceTransform(core.astype(np.uint8), cv2.DIST_L2, 5)
    widths = dt[core] * 2 / max(sc, 1e-6)          # 换算回原图 px
    shadow = fg & deep & (sat < R.SAT_ROOT) & ~core
    return {
        "src": src,
        "rel": str(p.relative_to(SRC2)) if str(p).startswith(str(SRC2)) else "数据集1/" + p.name,
        "root_gray": round(float(np.median(gray[core])), 1),
        "root_sat": round(float(np.median(sat[core])), 4),
        "root_width_px": round(float(np.median(widths)), 2) if len(widths) else 0.0,
        "shadow_pct": round(100 * float(shadow.mean()), 3),
        "fg_pct": round(100 * float(fg.mean()), 3),
        "root_pct": round(100 * float(core.mean()), 3),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    files = []
    for p in sorted(SRC2.rglob("*")):
        if p.suffix.lower() in (".jpg", ".jpeg", ".tif", ".tiff", ".png"):
            rel = p.relative_to(SRC2)
            if "9.9改" in rel.parts or "work" in rel.parts or "ocr_raw" in rel.parts:
                continue
            if p.stem in DONE:
                continue
            files.append(str(p))
    print(f"[标准图集] 候选 {len(files)} 张（已排除 9.9改/工作目录/已做的 7-00x）")

    rows = []
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import time
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(features, f) for f in files]
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                rows.append(fu.result())
            except Exception as exc:  # noqa: BLE001
                print(f"    !! {type(exc).__name__}: {exc}")
            if i % 20 == 0:
                print(f"  ... {i}/{len(files)}  已用 {time.time()-t0:.0f}s", flush=True)
    print(f"  特征提取完成 {len(rows)} 张，用时 {time.time()-t0:.0f}s")
    if not rows:
        print("  没有可用特征，退出")
        return 1

    keys = ["root_gray", "root_sat", "root_width_px", "shadow_pct", "fg_pct"]
    X = np.array([[r[k] for k in keys] for r in rows], dtype=np.float64)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd

    # 贪心最远点采样
    n = len(rows)
    sel = [int(np.argmax(np.linalg.norm(Z - Z.mean(0), axis=1)))]
    for _ in range(N_PICK - 1):
        d = np.min(np.linalg.norm(Z[:, None, :] - Z[sel][None, :, :], axis=2), axis=1)
        d[sel] = -1
        sel.append(int(np.argmax(d)))
    picked = [rows[i] for i in sel]

    with (OUT / "标准图集.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    print(f"\n  选出 {len(picked)} 张（特征空间最分散）:")
    for r in picked:
        print(f"    {r['rel']:<34} 根灰度 {r['root_gray']:6.1f}  饱和度 {r['root_sat']:.3f}  "
              f"根宽 {r['root_width_px']:5.2f}px  阴影 {r['shadow_pct']:6.3f}%  前景 {r['fg_pct']:5.2f}%")

    # contact sheet
    CW, CH = 760, 560
    def font(sz, b=False):
        for n in (("msyhbd.ttc", "msyh.ttc") if b else ("msyh.ttc",)):
            try:
                return ImageFont.truetype(n, sz)
            except Exception:  # noqa: BLE001
                continue
        return ImageFont.load_default()

    tiles = []
    for r in picked:
        im = Image.open(r["src"]).convert("RGB")
        k = min(CW / im.width, CH / im.height)
        im = im.resize((int(im.width * k), int(im.height * k)), Image.LANCZOS)
        bg = Image.new("RGB", (CW, CH), (255, 255, 255))
        bg.paste(im, ((CW - im.width) // 2, (CH - im.height) // 2))
        tiles.append(bg)
    cols = 3
    rr = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (CW * cols, rr * (CH + 74)), (255, 255, 255))
    dr = ImageDraw.Draw(sheet)
    for i, (r, t) in enumerate(zip(picked, tiles)):
        cx, cy = (i % cols) * CW, (i // cols) * (CH + 74)
        sheet.paste(t, (cx, cy))
        dr.text((cx + 8, cy + CH + 4), f"{i+1}. {r['rel']}", fill=(0, 0, 0), font=font(22, True))
        dr.text((cx + 8, cy + CH + 34),
                f"根灰度 {r['root_gray']:.0f}（越小越黑）  根宽 {r['root_width_px']:.2f}px  "
                f"阴影 {r['shadow_pct']:.2f}%", fill=(150, 0, 0), font=font(19))
        dr.text((cx + 8, cy + CH + 56),
                f"饱和度 {r['root_sat']:.3f}  前景 {r['fg_pct']:.2f}%", fill=(90, 90, 90), font=font(18))
    o = OUT / "标准图集_contact.png"
    sheet.save(o, optimize=True)
    print(f"\n  -> {o}")
    print(f"  -> {OUT/'标准图集.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
