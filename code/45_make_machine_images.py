"""用最优参数出机器图（二值化纯黑白 TIFF）+ 与人工图做像素级叠加对比。

最优参数（由 44_optimize_params.py 在 6 张人工真值上扫出）:
  HALO_GROW=0            贴根外扩彻底关掉（实测该参数无效）
  SHRINK_PX=1            收边 1px
  SMOOTH_OPEN_PX=1       去毛刺
  BRIDGE_CLOSE_PX=2      接断点
  -> 面积倍数 1.015 / 宽度比 1.041 / 长度比 0.981

二值化:
  人工图与机器图统一用阈值 210（该值由阈值敏感性分析选出：掩膜面积对阈值最不敏感处）
  灰噪提示: G218-001 的人工图灰噪最多(0.25%)，好在阈值 210 已在其上沿之下

产出:
  测试\\机器\\<样本>-mac.tif        纯黑白未压缩 TIFF，供 WinRHIZO 扫描
  outputs\\机器图对比\\<样本>_overlay.png   四格: 原图/人工二值/机器二值/叠加误差
  outputs\\机器图对比\\指标.csv
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
MAN = P.TEST_MAN
ORIG = P.TEST_ORIG
MAC = P.TEST_MAC
OUT = PROJ / "outputs" / "机器图对比"
SAMPLES = ["66-003", "28-CL-003", "G218-001", "126-003", "134-003", "7-003"]
THR = 210

# ---- 最优参数 ----
R.HALO_GROW = 0
R.SHRINK_PX = 1
R.SMOOTH_ALL = True
R.SMOOTH_OPEN_PX = 1
R.BRIDGE_CLOSE_PX = 2


def font(sz, b=False):
    for n in (("msyhbd.ttc", "msyh.ttc") if b else ("msyh.ttc",)):
        try:
            return ImageFont.truetype(n, sz)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


def orig_of(stem):
    for e in (".tif", ".jpg", ".png"):
        p = ORIG / (stem + e)
        if p.exists():
            return p
    return None


def panel(img, title, box=None, width=620, zoom=False):
    if box:
        x0, y0, x1, y1 = box
        img = img[y0:y1, x0:x1]
    if zoom:
        Z = max(1, int(width / img.shape[1]))
        disp = cv2.resize(img, None, fx=Z, fy=Z, interpolation=cv2.INTER_NEAREST)
    else:
        disp = cv2.resize(img, (width, int(img.shape[0] * width / img.shape[1])),
                          interpolation=cv2.INTER_AREA)
    c = Image.new("RGB", (disp.shape[1], disp.shape[0] + 40), (255, 255, 255))
    c.paste(Image.fromarray(disp), (0, 40))
    ImageDraw.Draw(c).text((8, 8), title, fill=(0, 0, 0), font=font(20))
    return c


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    MAC.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in SAMPLES:
        pm, po = MAN / f"{s}-man.tif", orig_of(s)
        if not pm.exists() or po is None:
            print(f"  跳过 {s}: 缺文件")
            continue
        rgb = np.asarray(Image.open(po).convert("RGB"))
        man = np.asarray(Image.open(pm).convert("L")) < THR
        root, _, info = R.extract_root(rgb, return_maps=False)
        if root.shape != man.shape:
            print(f"  跳过 {s}: 尺寸 {root.shape} vs {man.shape}")
            continue

        # 二值化机器图: 根=纯黑, 背景=纯白
        binimg = np.full(root.shape, 255, np.uint8)
        binimg[root] = 0
        dst = MAC / f"{s}-mac.tif"
        Image.fromarray(binimg).convert("RGB").save(
            dst, format="TIFF", compression="raw", dpi=(600, 600))

        inter = int((man & root).sum())
        iou = inter / max(1, int((man | root).sum()))
        rec = inter / max(1, int(man.sum()))
        pre = inter / max(1, int(root.sum()))
        rows.append({"样本": s, "阈值": THR,
                     "人工面积%": round(100 * man.mean(), 4),
                     "机器面积%": round(100 * root.mean(), 4),
                     "面积倍数": round(root.mean() / max(1e-9, man.mean()), 3),
                     "IoU": round(iou, 4), "召回": round(rec, 4), "精确": round(pre, 4),
                     "一致像素%": round(100 * inter / max(1, int((man | root).sum())), 2)})
        print(f"  {s:<12} 人工 {100*man.mean():6.3f}%  机器 {100*root.mean():6.3f}%  "
              f"倍数 {rows[-1]['面积倍数']:.3f}  IoU {iou:.4f}  召回 {rec:.4f}  精确 {pre:.4f}")

        # 叠加误差图
        vis = rgb.astype(np.float32)
        vis[man & root] = vis[man & root] * 0.35 + np.array([0, 175, 0]) * 0.65     # 一致=绿
        vis[root & ~man] = vis[root & ~man] * 0.2 + np.array([255, 40, 40]) * 0.8   # 机器多=红
        vis[man & ~root] = vis[man & ~root] * 0.2 + np.array([40, 90, 255]) * 0.8   # 人工多=蓝
        H, W = man.shape
        ys, xs = np.nonzero(man | root)
        if len(ys):
            cy, cx = int(np.median(ys)), int(np.median(xs))
            BW, BH = min(1100, W), min(800, H)
            x0 = int(np.clip(cx - BW // 2, 0, W - BW)); y0 = int(np.clip(cy - BH // 2, 0, H - BH))
            box = (x0, y0, x0 + BW, y0 + BH)
        else:
            box = None
        p4 = [panel(rgb, f"① 原图  {s}", box, zoom=True),
              panel(np.stack([np.where(man, 0, 255)] * 3, -1).astype(np.uint8),
                    "② 人工修图（二值）", box, zoom=True),
              panel(np.stack([binimg] * 3, -1).astype(np.uint8),
                    "③ 机器修图（二值）", box, zoom=True),
              panel(vis.astype(np.uint8),
                    f"④ 叠加  绿=一致 红=机器多 蓝=人工多   IoU {iou:.3f}", box, zoom=True)]
        h = max(p.height for p in p4)
        row = Image.new("RGB", (sum(p.width for p in p4), h), (255, 255, 255))
        x = 0
        for p in p4:
            row.paste(p, (x, 0)); x += p.width
        row.save(OUT / f"{s}_overlay.png", optimize=True)

    with (OUT / "指标.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    import statistics as st
    print(f"\n  中位: IoU {st.median(r['IoU'] for r in rows):.4f}  "
          f"召回 {st.median(r['召回'] for r in rows):.4f}  "
          f"精确 {st.median(r['精确'] for r in rows):.4f}  "
          f"面积倍数 {st.median(r['面积倍数'] for r in rows):.3f}")
    print(f"  机器图(二值TIFF) -> {MAC}")
    print(f"  叠加对比图 -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
