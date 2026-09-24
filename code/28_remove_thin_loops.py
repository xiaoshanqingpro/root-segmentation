"""白底根图后处理：剔除水渍反光形成的"细壁闭合环"。

背景（用户多次反馈）:
  水渍里的水泡轮廓，在用油漆桶/洪水填充把背景倒白之后会凸显出来，
  表现成**极细、浅色、成环或成弧**的曲线，常出现在分叉处。它们不是根。

前几轮失败的原因:
  在 1400~2600 px 的**特征尺度**上做检测 —— 细线被压成亚像素、断成上千碎片，
  "洞"根本形不成，判据失效。
本轮改为**直接在白底根图上、按原分辨率做后处理**：
  白底图里 非白像素 = 被判为根的像素，结构干净，不再有特征尺度的问题。

判据（本质区别）:
  根是实心树状结构，不会围出洞；
  水泡线是闭合细线，**内部是纯背景（白）**，而且**围壁极薄**。
  于是: 对根掩膜求"洞"，再用**逐级腐蚀**测围壁厚度 ——
        细线围的洞在半径 1~2 的腐蚀后就与外界连通（消失）；根围的洞不会。

用法:
    python 28_remove_thin_loops.py --in 白底根图 [--out 白底根图_净] [--dry-run]
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
import rootseg_common as R  # noqa: E402

# --- 统一路径（见 paths.py；数据搬家只改那里）---
sys.path.insert(0, str(Path(__file__).parent))
import paths as P  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
PROJ = P.PROJ

MAX_WALL_ERODE = 3      # 腐蚀到第几级就算"薄壁"（半径 px，原分辨率）
MIN_HOLE_PX = 400       # 洞面积下限，滤掉噪声
NONWHITE = 250          # >= 此值算白（背景）
WORKERS = 8


def find_thin_loops(p: Path):
    """返回 (需剔除的像素掩膜, 统计)"""
    rgb = np.asarray(Image.open(p).convert("RGB"))
    fg = ~np.all(rgb >= NONWHITE, axis=2)
    if not fg.any():
        return None, {}
    filled = ndimage.binary_fill_holes(fg)
    holes = filled & ~fg
    if not holes.any():
        return None, {"holes": 0, "thin": 0}

    n, lab, stats, _ = cv2.connectedComponentsWithStats(holes.astype(np.uint8), 8)
    big = [(j, stats[j, cv2.CC_STAT_AREA]) for j in range(1, n)
           if stats[j, cv2.CC_STAT_AREA] >= MIN_HOLE_PX]
    if not big:
        return None, {"holes": n - 1, "big_holes": 0, "thin": 0}

    ker = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    drop = np.zeros_like(fg)
    thin = 0
    for r in range(1, MAX_WALL_ERODE + 1):
        if not big:
            break
        er = cv2.erode(fg.astype(np.uint8), ker(r)) > 0
        holes_er = ndimage.binary_fill_holes(er) & ~er
        still = []
        for j, area in big:
            m = lab == j
            ys, xs = np.nonzero(m)
            y0, y1 = ys.min(), ys.max() + 1
            x0, x1 = xs.min(), xs.max() + 1
            # 该洞是否还在（腐蚀后仍有洞像素落在它的外接框内）
            if holes_er[y0:y1, x0:x1].any():
                still.append((j, area))
            else:
                # 壁太薄 -> 剔除紧邻该洞的根像素
                pad = r + 2
                yy0, yy1 = max(0, y0 - pad), min(fg.shape[0], y1 + pad)
                xx0, xx1 = max(0, x0 - pad), min(fg.shape[1], x1 + pad)
                near = m[yy0:yy1, xx0:xx1]
                sub = fg[yy0:yy1, xx0:xx1]
                drop[yy0:yy1, xx0:xx1] |= sub & cv2.dilate(near.astype(np.uint8), ker(pad)) > 0
                thin += 1
        big = still
    return drop, {"holes": n - 1, "big_holes": len(big), "thin": thin,
                  "drop_px": int(drop.sum())}


def worker(args):
    src, dst, make_out = args
    cv2.setNumThreads(1)
    t0 = time.time()
    src_p, dst_p = Path(src), Path(dst)
    drop, st = find_thin_loops(src_p)
    if make_out and drop is not None:
        rgb = np.asarray(Image.open(src_p).convert("RGB"))
        out = rgb if rgb.flags.writeable else rgb.copy()
        out[drop] = 255
        dst_p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(out).save(dst_p, "JPEG", quality=R.JPEG_Q, subsampling=0, dpi=(600, 600))
    return {"file": src_p.name, "rel": str(src_p), **st, "seconds": round(time.time() - t0, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=WORKERS)
    a = ap.parse_args()

    src = PROJ / a.src
    files = sorted(src.rglob("*.jpg"))
    print(f"[细壁闭合环剔除] 源 {src}  {len(files)} 张")
    print(f"[细壁闭合环剔除] 壁厚阈值: 腐蚀 <= {MAX_WALL_ERODE}px 即判薄壁   洞面积 >= {MIN_HOLE_PX}px")

    tasks = []
    for p in files:
        rel = p.relative_to(src)
        dst = (PROJ / a.dst / rel) if a.dst else p
        tasks.append((str(p), str(dst), not a.dry_run))

    rows = R.run_parallel(tasks, workers=a.workers, label="环剔除")
    hit = [r for r in rows if r.get("thin", 0) > 0]
    tot = sum(r.get("thin", 0) for r in rows)
    print(f"\n  检出细壁闭合环: {len(hit)}/{len(rows)} 张，共 {tot} 处")
    for r in sorted(hit, key=lambda z: -z.get("thin", 0))[:15]:
        print(f"    {r['file']:<22} 环 {r['thin']:3d} 处   剔除 {r.get('drop_px',0):>7} px")
    out_csv = PROJ / "outputs" / "细壁闭合环检测.csv"
    with out_csv.open("w", newline="", encoding="utf-8-sig") as fh:
        keys = sorted({k for r in rows for k in r})
        wr = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)
    print(f"  清单 -> {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
