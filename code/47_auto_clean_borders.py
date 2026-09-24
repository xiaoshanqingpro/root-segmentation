"""自动涂白扫描仪黑边（只针对人工标注图）。

目标文件: 22-001 / 44-001 / 46-002 的 *-man 图
问题: 这三张人工图没清扫描黑边，标签里黑边=根，导致 U-Net 把黑边学成根

策略（保守，避免误删真根）:
  1. 深色 = 灰度 < DARK_T
  2. 逐边检测"黑边带"宽度：从边缘往内扫，只要该行/列的深色贯通率 > COVER 就继续
  3. 候选 = 深色 ∩ 黑边带
  4. 再取候选的**连通块**，只保留"确实贴到图像最外缘"的那些  <- 关键：不贴边的深色不动
  5. 把这些像素涂白

不会动的东西: 没伸到最外缘的根系、浅灰的根缘、图内任何深色根

产出: outputs/边框清理/cleaned/<样本>-man-clean.tif
      outputs/边框清理/<样本>_before_after.png
      outputs/边框清理/border_clean_report.csv
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
import paths as P  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
OUT = P.PROJ / "outputs" / "边框清理"
CLEAN_DIR = OUT / "cleaned"
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = ["22-001", "44-001", "46-002"]
DARK_T = 110        # 灰度低于此值算"深色"
COVER = 0.35        # 该行/列深色贯通率超过此值，认为还在黑边带里
MAX_BAND = 500      # 单边最大带宽（防意外）
NONWHITE = 250

# 第二遍：浅色纸边 / 盖板边（深色黑边清完之后剩下的那层米黄色边条）
# 判据：贴图像最外缘 且 外接框在某个方向跨越大半个画幅 且 整体偏亮
# 依据：纸边/盖板边是浅米色（均值灰度高），而根是深色组织；且根不会既贴边又横跨半张图
LIGHT_SPAN = 0.40
LIGHT_GRAY_MIN = 140


def font(sz):
    for n in ("msyh.ttc", "simhei.ttf"):
        try:
            return ImageFont.truetype(n, sz)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


def find_man(stem: str) -> Path | None:
    for d in [P.TEST_MAN_99GAI, P.TEST_MAN]:
        for q in sorted(d.glob("*")):
            if "-man" in q.stem.lower() and \
                    q.stem.replace("-man", "").replace("-9.22", "") == stem:
                return q
    return None


def band_profile(dark: np.ndarray, axis: int, reverse: bool) -> int:
    """从边缘往里扫，返回黑边带宽度"""
    prof = dark.mean(axis=1) if axis == 0 else dark.mean(axis=0)
    if reverse:
        prof = prof[::-1]
    w = 0
    for i in range(min(MAX_BAND, len(prof))):
        if prof[i] > COVER:
            w = i + 1
        elif i > 4 and prof[i] <= COVER and prof[min(i + 3, len(prof) - 1)] <= COVER:
            break
    return w


def main() -> int:
    rows = []
    for stem in TARGETS:
        p = find_man(stem)
        if p is None:
            print(f"  !! {stem} 找不到")
            continue
        rgb = np.asarray(Image.open(p).convert("RGB")).copy()
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        H, W = gray.shape
        dark = gray < DARK_T

        top = band_profile(dark, 0, False)
        bot = band_profile(dark, 0, True)
        left = band_profile(dark, 1, False)
        right = band_profile(dark, 1, True)
        print(f"\n=== {stem} ({W}x{H}) ===")
        print(f"  黑边带宽度: 上 {top}  下 {bot}  左 {left}  右 {right}")

        band = np.zeros_like(dark)
        if top:
            band[:top, :] = True
        if bot:
            band[-bot:, :] = True
        if left:
            band[:, :left] = True
        if right:
            band[:, -right:] = True

        cand = dark & band
        # 只保留"贴到图像最外缘"的连通块
        n, lab, stats, _ = cv2.connectedComponentsWithStats(cand.astype(np.uint8), 8)
        on_edge = np.unique(np.concatenate([
            lab[0, :], lab[H - 1, :], lab[:, 0], lab[:, W - 1]]))
        keep = np.zeros(n, bool)
        keep[on_edge] = True
        keep[0] = False
        remove = keep[lab]

        before = np.any(rgb < NONWHITE, axis=2)
        out = rgb.copy()
        out[remove] = 255

        # ---- 第二遍：浅色纸边 / 盖板边 ----
        g2 = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
        nw2 = np.any(out < NONWHITE, axis=2)
        n2, l2, s2, _ = cv2.connectedComponentsWithStats(nw2.astype(np.uint8), 8)
        light_remove = np.zeros_like(nw2)
        for j in range(1, n2):
            x, y, w_, h_, area = s2[j, cv2.CC_STAT_LEFT], s2[j, cv2.CC_STAT_TOP], \
                s2[j, cv2.CC_STAT_WIDTH], s2[j, cv2.CC_STAT_HEIGHT], s2[j, cv2.CC_STAT_AREA]
            touches = (x <= 2 or y <= 2 or x + w_ >= W - 2 or y + h_ >= H - 2)
            if not touches:
                continue
            wide = (w_ >= LIGHT_SPAN * W) or (h_ >= LIGHT_SPAN * H)
            if not wide:
                continue
            m = l2 == j
            if float(g2[m].mean()) < LIGHT_GRAY_MIN:
                continue          # 整体偏暗 -> 可能是根，不动
            light_remove |= m
        n_light = int(light_remove.sum())
        out[light_remove] = 255
        after = np.any(out < NONWHITE, axis=2)

        d_before = int(before.sum())
        d_after = int(after.sum())
        print(f"  第一遍(深色黑边): {int(keep.sum())} 块 / {int(remove.sum()):,} px")
        print(f"  第二遍(浅色纸边): {n_light:,} px")
        print(f"  非白像素: {d_before:,} -> {d_after:,}  "
              f"(减少 {100*(1-d_after/max(1,d_before)):.2f}%)")
        # 边框带残留
        b80 = np.zeros_like(before)
        b80[:80, :] = b80[-80:, :] = b80[:, :80] = b80[:, -80:] = True
        print(f"  边框 80px 带非白: {100*before[b80].mean():.1f}% -> "
              f"{100*after[b80].mean():.1f}%")

        dst = CLEAN_DIR / f"{stem}-man-clean.tif"
        Image.fromarray(out).save(dst, dpi=(600, 600))
        rows.append({"样本": stem, "源文件": str(p), "输出": str(dst),
                     "带宽_上": top, "带宽_下": bot, "带宽_左": left, "带宽_右": right,
                     "剔除px": int(remove.sum()) + n_light,
                     "非白_before": d_before, "非白_after": d_after,
                     "边框带非白_before%": round(100 * before[b80].mean(), 2),
                     "边框带非白_after%": round(100 * after[b80].mean(), 2)})

        # 前后对比图（缩略 + 两处局部）
        def tile(arr, title, box=None, w=700):
            a = arr[box[1]:box[3], box[0]:box[2]] if box else arr
            im = Image.fromarray(a)
            k = w / im.width
            im = im.resize((w, max(1, int(im.height * k))), Image.LANCZOS)
            c = Image.new("RGB", (w, im.height + 34), (255, 255, 255))
            c.paste(im, (0, 34))
            ImageDraw.Draw(c).text((6, 6), title, fill=(0, 0, 0), font=font(19))
            return c

        top_tiles = [tile(rgb, "① 原人工图（黑边未清）"), tile(out, "② 自动涂白后")]
        # 顶部黑边局部
        b1 = (0, 0, 1600, 500)
        b2 = (W - 900, H // 2 - 400, W, H // 2 + 400)
        mid = [tile(rgb, "① 顶部黑边（原始）", b1, 780), tile(out, "② 顶部（已清）", b1, 780)]
        mid2 = [tile(rgb, "① 右侧黑边（原始）", b2, 780), tile(out, "② 右侧（已清）", b2, 780)]
        def hstack(ts):
            h = max(t.height for t in ts)
            im = Image.new("RGB", (sum(t.width for t in ts), h), (255, 255, 255))
            x = 0
            for t in ts:
                im.paste(t, (x, 0)); x += t.width
            return im
        r1, r2, r3 = hstack(top_tiles), hstack(mid), hstack(mid2)
        canvas = Image.new("RGB", (max(r1.width, r2.width, r3.width),
                                   r1.height + r2.height + r3.height + 24), (255, 255, 255))
        y = 0
        for r in (r1, r2, r3):
            canvas.paste(r, (0, y)); y += r.height + 12
        o = OUT / f"{stem}_before_after.png"
        canvas.save(o, optimize=True)
        print(f"  对比图 -> {o}")

    if rows:
        with (OUT / "border_clean_report.csv").open("w", newline="", encoding="utf-8-sig") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
    print(f"\n  清理后文件 -> {CLEAN_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
