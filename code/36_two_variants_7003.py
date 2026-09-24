"""出两档成品: 收边1px（保细根）与 收边2px（面积最贴真值），供用户/老师选。

收边是全局腐蚀，因此存在权衡:
  收边 1px -> 面积 1.33x，召回 0.891（细侧根基本保住）
  收边 2px -> 面积 1.09x，召回 0.816（面积最贴，但末端细根被腐蚀掉）
两档都清掉细壁闭合环，输出 PNG + 高质 JPEG + 并排对比图。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
import rootseg_common as R  # noqa: E402
import importlib.util as _il

# --- 统一路径（见 paths.py；数据搬家只改那里）---
sys.path.insert(0, str(Path(__file__).parent))
import paths as P  # noqa: E402

spec = _il.spec_from_file_location("fin", str(Path(__file__).parent / "35_finalize_7003.py"))
fin = _il.module_from_spec(spec)

Image.MAX_IMAGE_PIXELS = None
PROJ = P.PROJ
ORIG = P.RAW_D2_99 / "7-003.tif"
GT = P.TEST_MAN_99GAI / "7-003.tif"
OUT = PROJ / "outputs" / "7-003三方对比"
NONWHITE = 250


def font(sz):
    for n in ("msyh.ttc", "simhei.ttf"):
        try:
            return ImageFont.truetype(n, sz)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


def _remove_loops(rgb):
    """细壁闭合环清理：细壁围出的洞在逐级腐蚀后会消失，据此定位水泡细线。"""
    from scipy import ndimage
    fg = np.any(rgb < NONWHITE, axis=2)
    H, W = fg.shape
    ker = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    filled = ndimage.binary_fill_holes(fg)
    holes = filled & ~fg
    drop = np.zeros_like(fg)
    if not holes.any():
        return drop, 0
    nh, lh, sh, _ = cv2.connectedComponentsWithStats(holes.astype(np.uint8), 8)
    big = [(j, sh[j, cv2.CC_STAT_AREA]) for j in range(1, nh) if sh[j, cv2.CC_STAT_AREA] >= 300]
    cnt = 0
    for r in range(1, 4):
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
    return drop, cnt


def main() -> int:
    remove_loops = _remove_loops

    orig = np.asarray(Image.open(ORIG).convert("RGB")).copy()
    gt = np.any(np.asarray(Image.open(GT).convert("RGB")) < NONWHITE, axis=2)
    R.HALO_GROW = 0

    outs = {}
    for k in (1, 2):
        R.SHRINK_PX = k
        root, _, _ = R.extract_root(orig, return_maps=False)
        o = orig.copy()
        o[~root] = 255
        drop, nl = remove_loops(o)
        o[drop] = 255
        m = np.any(o < NONWHITE, axis=2)
        inter = int((gt & m).sum())
        outs[k] = {"img": o, "mask": m, "loops": nl, "drop": int(drop.sum()),
                   "IoU": inter / max(1, int((gt | m).sum())),
                   "recall": inter / max(1, int(gt.sum())),
                   "precision": inter / max(1, int(m.sum())),
                   "ratio": m.mean() / gt.mean()}
        Image.fromarray(o).save(OUT / f"7-003_final_shrink{k}.jpg", "JPEG", quality=98,
                                subsampling=0, dpi=(600, 600))
        Image.fromarray(o).save(OUT / f"7-003_final_shrink{k}.png", optimize=True)
        r = outs[k]
        print(f"  收边{k}px: IoU {r['IoU']:.4f}  召回 {r['recall']:.4f}  "
              f"精确 {r['precision']:.4f}  面积 {r['ratio']:.3f}x  清细线 {r['loops']} 处 "
              f"{r['drop']:,}px")

    # 并排对比: 真值 | 收边1 | 收边2 | 误差(收边1) | 误差(收边2)
    def err(m):
        ov = orig.astype(np.float32)
        ov[gt & m] = ov[gt & m] * 0.35 + np.array([0, 170, 0]) * 0.65
        ov[~gt & m] = ov[~gt & m] * 0.2 + np.array([255, 40, 40]) * 0.8
        ov[gt & ~m] = ov[gt & ~m] * 0.2 + np.array([40, 90, 255]) * 0.8
        return ov.astype(np.uint8)

    panels = [np.stack([gt * 255] * 3, -1).astype(np.uint8),
              np.full_like(orig, 255), np.full_like(orig, 255), err(outs[1]["mask"]), err(outs[2]["mask"])]
    panels[1][outs[1]["mask"]] = orig[outs[1]["mask"]]
    panels[2][outs[2]["mask"]] = orig[outs[2]["mask"]]
    titles = ["① 你精修的真值", "② 我·收边1px（保细根）", "③ 我·收边2px（面积最贴）",
              "④ 收边1px 误差", "⑤ 收边2px 误差"]
    PW = 640
    tiles = [cv2.resize(p, (PW, int(p.shape[0] * PW / p.shape[1])), interpolation=cv2.INTER_AREA)
             for p in panels]
    row = np.hstack(tiles)
    c = Image.new("RGB", (row.shape[1], row.shape[0] + 48), (255, 255, 255))
    c.paste(Image.fromarray(row), (0, 48))
    dr = ImageDraw.Draw(c)
    dr.text((8, 6), f"7-003 两档成品对比   真值面积 {100*gt.mean():.3f}%   "
                    f"收边1: {outs[1]['ratio']:.2f}x 收边2: {outs[2]['ratio']:.2f}x   "
                    f"（误差图: 红=误检 蓝=漏检）", fill=(0, 0, 0), font=font(24))
    for i, t in enumerate(titles):
        dr.text((8 + i * PW, 30), t, fill=(150, 0, 0), font=font(18))
    o = OUT / "7-003_两档对比.png"
    c.save(o, optimize=True)
    print(f"  -> {o}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
