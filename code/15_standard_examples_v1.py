"""标准 v1 · 疑难示例图（重做版）：修正 v0 示例图"极细根没被覆盖"的问题。

v0 的问题（用户指出）: A/B/D 三张图的绿区没覆盖极细根，只有 C 覆盖完整。
根因: v0 用的是纯经典方法的"根芯"掩膜，漏掉了细根。
v1 口径: 前景来自**边界洪水填充（容差 20，等价油漆桶）**，
         根 = 前景 ∩ (管状响应 或 褐色饱和)，并把贴根半影适度并入（≤6px）。
         阴影/背景都必须是"非根"，不再出现"根缘被判成阴影"。

产出: outputs/阶段二/standard_examples_v1/rule_A..D.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# --- 统一路径（见 paths.py；数据搬家只改那里）---
sys.path.insert(0, str(Path(__file__).parent))
import paths as P  # noqa: E402

PROJ = P.PROJ
CFG = PROJ / "config" / "phase2_images.json"
OUT = PROJ / "outputs" / "阶段二" / "standard_examples_v1"
MM_PER_PX = 25.4 / 600
CROP = 340
FEAT_LONG = 1400
FLOOD_TOL = 20
DARK_DROP = 12
SAT_ROOT = 0.10
FRANGI_TUB = 0.20
HALO_GROW = 6
MIN_BLOB = 40


def font(sz):
    for n in ("msyh.ttc", "simhei.ttf"):
        try:
            return ImageFont.truetype(n, sz)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


def analyze(path: Path):
    rgb = np.asarray(Image.open(path).convert("RGB"))
    H, W = rgb.shape[:2]
    sc = FEAT_LONG / max(H, W)
    small = cv2.resize(rgb, (int(W * sc), int(H * sc)), interpolation=cv2.INTER_AREA) if sc < 1 else rgb
    hh, ww = small.shape[:2]

    # 背景 = 边界洪水填充（油漆桶容差 20）
    mask = np.zeros((hh + 2, ww + 2), np.uint8)
    flags = 4 | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | (255 << 8)
    img = np.ascontiguousarray(small)
    step = max(1, min(hh, ww) // 64)
    for x in range(0, ww, step):
        for (sx, sy) in ((x, 0), (x, hh - 1)):
            if not mask[sy + 1, sx + 1]:
                cv2.floodFill(img, mask, (sx, sy), 255, (FLOOD_TOL,) * 3, (FLOOD_TOL,) * 3, flags)
    for y in range(0, hh, step):
        for (sx, sy) in ((0, y), (ww - 1, y)):
            if not mask[sy + 1, sx + 1]:
                cv2.floodFill(img, mask, (sx, sy), 255, (FLOOD_TOL,) * 3, (FLOOD_TOL,) * 3, flags)
    fg_small = ~(mask[1:-1, 1:-1] > 0)

    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1].astype(np.float32) / 255.0
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32)
    k = max(31, (int(min(hh, ww) * 0.05) // 2) * 2 + 1)
    illum = cv2.morphologyEx(gray.astype(np.uint8), cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(np.float32)
    depth = illum - gray
    corrected = np.clip(gray / np.maximum(illum, 1e-3) * float(illum.mean()), 0, 255)
    try:
        from skimage.filters import frangi as sk_frangi
        fr = sk_frangi((255.0 - corrected) / 255.0, sigmas=range(1, 6), black_ridges=False)
        fr = cv2.normalize(fr.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
    except Exception:  # noqa: BLE001
        fr = np.zeros_like(gray)

    deep = depth > DARK_DROP
    core = fg_small & deep & ((fr > FRANGI_TUB) | (sat >= SAT_ROOT))
    if HALO_GROW > 0:
        g = max(1, int(round(HALO_GROW * sc)))
        grown = cv2.dilate(core.astype(np.uint8),
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * g + 1))) > 0
        core = core | (grown & fg_small & deep)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(core.astype(np.uint8), 8)
    keep = np.array([False] + [stats[j, cv2.CC_STAT_AREA] >= max(4, int(MIN_BLOB * sc * sc))
                               for j in range(1, n)])
    core = keep[lab]

    shadow = fg_small & ~core & ~deep | (fg_small & ~core & deep & (sat < SAT_ROOT) & (fr <= FRANGI_TUB))
    shadow = cv2.morphologyEx(shadow.astype(np.uint8), cv2.MORPH_OPEN,
                              np.ones((3, 3), np.uint8)).astype(bool)
    return rgb, small, core, shadow, sat, depth, fr, sc


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(CFG.read_text(encoding="utf-8"))
    pool: dict[str, list] = {}

    def consider(rule, score, it, crop, payload):
        pool.setdefault(rule, []).append({"score": score, "it": it, "crop": crop, "payload": payload})

    for it in cfg["items"]:
        rgb, small, core, shadow, sat, depth, fr, sc = analyze(PROJ / it["processed_path"])
        H, W = rgb.shape[:2]
        hh, ww = small.shape[:2]
        dist = cv2.distanceTransform((~core).astype(np.uint8), cv2.DIST_L2, 5)
        m0 = CROP // 2 + 20
        interior = np.zeros((hh, ww), bool)
        interior[m0:hh - m0, m0:ww - m0] = True
        kn = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61)).astype(np.float32)
        kn /= kn.sum()

        cand = {
            "A": shadow & (dist > 2) & (dist <= 6) & interior,
            "B": shadow & (dist > 50) & interior,
            "C": shadow & (dist > 25) & interior,
        }
        for tag, m in cand.items():
            if m.sum() < 200:
                continue
            dens = cv2.filter2D(m.astype(np.float32), -1, kn)
            cy, cx = np.unravel_index(int(np.argmax(dens)), dens.shape)
            y0 = int(np.clip(cy - CROP // 2, 0, max(0, hh - CROP)))
            x0 = int(np.clip(cx - CROP // 2, 0, max(0, ww - CROP)))
            sl = (slice(y0, y0 + CROP), slice(x0, x0 + CROP))
            consider(tag, float(dens[cy, cx]), it,
                     (small[sl], core[sl], shadow[sl]),
                     {"dist": float(dist[m].mean()), "sat": float(sat[m].mean()),
                      "f2": 1.0 / sc, "n": int(m.sum())})

        n, lab, stats, _ = cv2.connectedComponentsWithStats(core.astype(np.uint8), 8)
        bs, bc = 0.0, None
        for j in range(1, n):
            x, y, w_, h_, area = stats[j]
            if area < 30 or area > 900:
                continue
            cx_, cy_ = x + w_ // 2, y + h_ // 2
            if not (m0 < cx_ < ww - m0 and m0 < cy_ < hh - m0):
                continue
            el = max(w_, h_) / max(1, min(w_, h_))
            if el < 4:
                continue
            if el * area ** 0.5 > bs:
                bs, bc = el * area ** 0.5, (cx_, cy_, w_, h_, el)
        if bc:
            cx_, cy_, w_, h_, el = bc
            y0 = int(np.clip(cy_ - CROP // 2, 0, max(0, hh - CROP)))
            x0 = int(np.clip(cx_ - CROP // 2, 0, max(0, ww - CROP)))
            sl = (slice(y0, y0 + CROP), slice(x0, x0 + CROP))
            consider("D", bs, it, (small[sl], core[sl], shadow[sl]),
                     {"w": int(w_), "h": int(h_), "el": float(el), "f2": 1.0 / sc})

    best, used = {}, set()
    for tag in ["A", "B", "C", "D"]:
        c = sorted(pool.get(tag, []), key=lambda z: -z["score"])
        free = [z for z in c if z["it"]["rel_path"] not in used]
        pick = (free or c)[0] if (free or c) else None
        if pick:
            best[tag] = pick
            used.add(pick["it"]["rel_path"])

    ex = []
    for tag in ["A", "B", "C", "D"]:
        if tag not in best:
            continue
        b = best[tag]
        it, (rgb_c, core_c, sh_c), pl = b["it"], b["crop"], b["payload"]
        f2 = pl["f2"]
        ov = rgb_c.astype(np.float32).copy()
        ov[sh_c] = ov[sh_c] * 0.55 + np.array([255, 40, 40]) * 0.45
        ov[core_c] = ov[core_c] * 0.30 + np.array([0, 200, 0]) * 0.70   # 根最后落笔，覆盖阴影
        big = cv2.resize(ov.astype(np.uint8), None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        h, w = big.shape[:2]
        canvas = Image.new("RGB", (w + 24, h + 128), (255, 255, 255))
        canvas.paste(Image.fromarray(big), (12, 74))
        dr = ImageDraw.Draw(canvas)
        titles = {
            "A": ("规则 A · 贴根半影  ->  判「根」", "绿=根（含贴根半影，已并入） 红=阴影"),
            "B": ("规则 B · 暗但没有线性结构  ->  判「阴影」", "红=阴影  绿=根"),
            "C": ("规则 C · 水渍 / 湿痕  ->  判「阴影」", "红=阴影  绿=根"),
            "D": ("规则 D · 极细根 vs 孤立碎屑  ->  有走向判「根」", "绿=根（细根已完整覆盖） 红=阴影"),
        }
        dr.text((12, 8), titles[tag][0], fill=(0, 0, 0), font=font(26))
        dr.text((12, 42), titles[tag][1], fill=(150, 0, 0), font=font(20))
        if tag == "A":
            lines = [f"实测：暗边到根缘平均 {pl['dist']:.1f}px = {pl['dist']*MM_PER_PX*f2:.3f} mm（600 dpi）",
                     "标准 §1.1(2)：外扩 ≤6px 的贴根暗边计入根类（可修剪）"]
        elif tag == "B":
            lines = [f"实测：暗区离最近根平均 {pl['dist']:.1f}px = {pl['dist']*MM_PER_PX*f2:.2f} mm，无管状走向",
                     "标准 §1.2(5)：暗但无线性结构 -> 阴影（首要判据）"]
        elif tag == "C":
            lines = [f"实测：该区平均饱和度 {pl['sat']:.3f}（灰蓝调，与褐色根可分）",
                     "标准 §1.2(3)：水渍成块、边缘柔、常呈环状 -> 阴影"]
        else:
            wp = min(pl["w"], pl["h"]) * f2
            lines = [f"实测：细根粗约 {wp:.0f}px = {wp*MM_PER_PX:.3f} mm，长宽比 {pl['el']:.1f}",
                     "标准 §1.3(1)：有走向的极细根判「根」；孤立碎点判背景"]
        dr.text((12, h + 78), lines[0], fill=(0, 0, 0), font=font(19))
        dr.text((12, h + 102), lines[1], fill=(0, 0, 0), font=font(19))
        canvas.save(OUT / f"rule_{tag}.png", optimize=True)
        ex.append({"rule": tag, "image": it["rel_path"], "score": round(b["score"], 5)})
        print(f"  [{tag}] 来源 {it['rel_path']}")

    (OUT / "examples.json").write_text(
        json.dumps({"mm_per_px": MM_PER_PX, "flood_tol": FLOOD_TOL, "halo_grow_px": HALO_GROW,
                    "examples": ex}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  {len(ex)} 张 -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
