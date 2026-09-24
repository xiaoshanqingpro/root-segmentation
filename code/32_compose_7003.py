"""7-003 三方对比图：原图 / 用户精修图 / 用户在精修基础上再经我优化后的图。

用户说明: "7001 7002 7003 是精修过的图片，其中 7003 是修得最仔细的一张"。
所以 7-003 是当前最可信的一份人工结果，用它做样本。

"我的优化" = 在**用户精修图**的基础上做一次收尾清理（不重新判根），针对用户指出的残留:
  1. 未清理的扫描边框/黑边（贴边且跨度大，或贴边且整体偏亮）
  2. 水渍反光形成的**细壁闭合环**（用逐级腐蚀测壁厚: 细壁围的洞在半径<=3 的腐蚀后消失）
  3. 无彩的浅灰块（块均灰度偏高且不细长）

产出: outputs/7-003三方对比/7-003_compare.png   （2 行 x 3 列：全图 + 局部放大）
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

# --- 统一路径（见 paths.py；数据搬家只改那里）---
sys.path.insert(0, str(Path(__file__).parent))
import paths as P  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
PROJ = P.PROJ
ORIG = P.RAW_D2_99 / "7-003.tif"
USER = P.TEST_MAN_99GAI / "7-003.tif"
OUT = PROJ / "outputs" / "7-003三方对比"

NONWHITE = 250
EDGE_SPAN = 0.25
EDGE_LIGHT = 195
MAX_WALL_ERODE = 3
MIN_HOLE_PX = 300
GRAY_COMP_MAX = 200
SAT_COMP_MIN = 0.05
ELONG_MIN = 4.0


def font(sz):
    for n in ("msyh.ttc", "simhei.ttf"):
        try:
            return ImageFont.truetype(n, sz)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


def cleanup(rgb: np.ndarray):
    """在用户精修图上做收尾清理，返回 (清理后rgb, 各部分掩膜, 统计)"""
    fg = np.any(rgb < NONWHITE, axis=2)
    H, W = fg.shape
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1].astype(np.float32) / 255.0
    ker = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))

    m_edge = np.zeros_like(fg)
    n, lab, st, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), 8)
    for j in range(1, n):
        x, y, w, h, a = st[j]
        touches = (x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1)
        if not touches:
            continue
        if (w >= EDGE_SPAN * W) or (h >= EDGE_SPAN * H) or (float(gray[lab == j].mean()) > EDGE_LIGHT):
            m_edge |= (lab == j)

    m_loop = np.zeros_like(fg)
    rest = fg & ~m_edge
    filled = ndimage.binary_fill_holes(rest)
    holes = filled & ~rest
    if holes.any():
        nh, lh, sh, _ = cv2.connectedComponentsWithStats(holes.astype(np.uint8), 8)
        big = [(j, sh[j, cv2.CC_STAT_AREA]) for j in range(1, nh) if sh[j, cv2.CC_STAT_AREA] >= MIN_HOLE_PX]
        for r in range(1, MAX_WALL_ERODE + 1):
            if not big:
                break
            er = cv2.erode(rest.astype(np.uint8), ker(r)) > 0
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
                    sub = rest[yy0:yy1, xx0:xx1]
                    m_loop[yy0:yy1, xx0:xx1] |= sub & (cv2.dilate(near.astype(np.uint8), ker(pad)) > 0)
            big = still

    m_gray = np.zeros_like(fg)
    rest2 = rest & ~m_loop
    ng, lg, sg, _ = cv2.connectedComponentsWithStats(rest2.astype(np.uint8), 8)
    for j in range(1, ng):
        x, y, w, h, a = sg[j]
        m = lg == j
        if float(gray[m].mean()) <= GRAY_COMP_MAX:
            continue
        if float(sat[m].mean()) >= SAT_COMP_MIN:
            continue
        sub = m[y:y + h, x:x + w]
        half_w = float(cv2.distanceTransform(sub.astype(np.uint8), cv2.DIST_L2, 5).max())
        elong = (a ** 0.5) / max(0.5, half_w)
        if elong >= ELONG_MIN:      # 细长的浅色结构保留（可能是浅色细根）
            continue
        m_gray |= m

    drop = m_edge | m_loop | m_gray
    out = rgb.copy()
    out[drop] = 255
    stat = {"edge_px": int(m_edge.sum()), "loop_px": int(m_loop.sum()),
            "grayblob_px": int(m_gray.sum()), "total_drop": int(drop.sum()),
            "fg_before": int(fg.sum()), "fg_after": int((fg & ~drop).sum())}
    return out, drop, m_loop, stat


def tile(img: np.ndarray, title: str, box=None, width=660, zoom=False):
    if box:
        x0, y0, x1, y1 = box
        img = img[y0:y1, x0:x1]
    if zoom:
        Z = max(1, int(width / img.shape[1]))
        img = cv2.resize(img, None, fx=Z, fy=Z, interpolation=cv2.INTER_NEAREST)
        disp = img
    else:
        disp = cv2.resize(img, (width, int(img.shape[0] * width / img.shape[1])),
                          interpolation=cv2.INTER_AREA)
    c = Image.new("RGB", (disp.shape[1], disp.shape[0] + 40), (255, 255, 255))
    c.paste(Image.fromarray(disp), (0, 40))
    ImageDraw.Draw(c).text((8, 8), title, fill=(0, 0, 0), font=font(22))
    return c


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("读取原图 …")
    orig = np.asarray(Image.open(ORIG).convert("RGB")).copy()
    print("读取你精修的图 …")
    user = np.asarray(Image.open(USER).convert("RGB")).copy()
    print("执行收尾清理 …")
    mine, drop, m_loop, st = cleanup(user)
    print(f"  扫描边框类 {st['edge_px']:,} px   细壁闭合环 {st['loop_px']:,} px   "
          f"无彩浅灰块 {st['grayblob_px']:,} px")
    print(f"  前景 {st['fg_before']:,} -> {st['fg_after']:,} px "
          f"（减少 {100*(1-st['fg_after']/max(1,st['fg_before'])):.2f}%）")

    # 局部放大：选"细壁闭合环"最集中的区域
    if m_loop.any():
        ys, xs = np.nonzero(m_loop)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (301, 301))
        dens = cv2.filter2D(m_loop.astype(np.float32), -1, k / k.sum())
        cy, cx = np.unravel_index(int(np.argmax(dens)), dens.shape)
    else:
        cy, cx = orig.shape[0] // 2, orig.shape[1] // 2
    BW, BH = 1100, 780
    H, W = orig.shape[:2]
    x0 = int(np.clip(cx - BW // 2, 0, W - BW)); y0 = int(np.clip(cy - BH // 2, 0, H - BH))
    box = (x0, y0, x0 + BW, y0 + BH)
    print(f"  局部放大区域: ({x0},{y0})-({x0+BW},{y0+BH})")

    # 误差着色图（= 你精修图，把我剔除的标红）
    ov = user.astype(np.float32)
    ov[drop] = ov[drop] * 0.25 + np.array([255, 40, 40]) * 0.75
    ov = ov.astype(np.uint8)

    row1 = [tile(orig, "① 原图（9.9\\7-003.tif）"),
            tile(user, "② 你精修的图（9.9改\\7-003.tif）"),
            tile(ov, "③ 你精修 + 我优化后（红=我剔除的部分）")]
    row1 = [tile(orig, "① 原图（9.9\\7-003.tif）", box, zoom=True),
            tile(user, "② 你精修的图（9.9改\\7-003.tif）", box, zoom=True),
            tile(ov, "③ 你精修 + 我优化后（红=我剔除的）", box, zoom=True)]
    top = [tile(orig, "① 原图", None, 660), tile(user, "② 你精修的图", None, 660),
           tile(mine, "③ 你精修 + 我优化后", None, 660)]
    h1 = max(t.height for t in top)
    left = Image.new("RGB", (sum(t.width for t in top), h1), (255, 255, 255))
    xx = 0
    for t in top:
        left.paste(t, (xx, 0)); xx += t.width
    h2 = max(t.height for t in row1)
    right = Image.new("RGB", (sum(t.width for t in row1), h2), (255, 255, 255))
    xx = 0
    for t in row1:
        right.paste(t, (xx, 0)); xx += t.width
    Wc = max(left.width, right.width)
    canvas = Image.new("RGB", (Wc, left.height + right.height + 20), (255, 255, 255))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (0, left.height + 20))
    o = OUT / "7-003_compare.png"
    canvas.save(o, optimize=True)
    print(f"\n  -> {o}  {canvas.size}")

    # 单独存一份"我优化后的图"
    Image.fromarray(mine).save(OUT / "7-003_optimized.jpg", "JPEG", quality=95,
                               subsampling=0, dpi=(600, 600))
    print(f"  -> {OUT/'7-003_optimized.jpg'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
