"""阶段二 · 为《阴影标注判定标准》生成疑难示例图（像素级实证）。

标准里每条规则都该有"长什么样"的样例。本脚本自动定位 4 类边界情形，裁 1:1 像素
放大图，叠加我建议的标注（绿=根 / 红=阴影），并标注实测宽度（px 与 mm），供人复核。

4 类样例:
  A. 贴根半影      -> 标准 §1.1(2)：紧贴根、宽度 ≤6px 的柔和暗边，建议计入"根"
  B. 暗但无线性结构 -> 标准 §1.2(5)：与根无关的大片暗区（褶皱/照明不均），建议判"阴影"
  C. 水渍/湿痕      -> 标准 §1.2(3)：低饱和成块、边缘柔，建议判"阴影"
  D. 极细根 vs 碎屑 -> 标准 §1.3(1)：细到 1-3px；成走向的判"根"，孤立碎点判"背景"

产出: outputs/阶段二/standard_examples/rule_A..D.png  +  examples.json
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
IMAGES = PROJ / "processed" / "600dpi"
CFG = PROJ / "config" / "phase2_images.json"
OUT = PROJ / "outputs" / "阶段二" / "standard_examples"
MM_PER_PX = 25.4 / 600
CROP = 340          # 1:1 像素裁剪边长
WORK = 1600


def font(sz):
    for n in ("msyh.ttc", "simhei.ttf"):
        try:
            return ImageFont.truetype(n, sz)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


def layers(path: Path):
    im = Image.open(path)
    im.draft("RGB", (WORK, WORK))
    im = im.convert("RGB")
    s = WORK / max(im.size)
    if s < 1:
        im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
    rgb = np.asarray(im)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1].astype(np.float32) / 255.0
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))
    illum = cv2.morphologyEx(gray.astype(np.uint8), cv2.MORPH_CLOSE, ker).astype(np.float32)
    depth = illum - gray
    corrected = np.clip(gray / np.maximum(illum, 1e-3) * float(illum.mean()), 0, 255)
    inv = (255.0 - corrected) / 255.0
    try:
        from skimage.filters import frangi as sk_frangi
        fr = sk_frangi(inv, sigmas=range(1, 7), black_ridges=False)
        fr = cv2.normalize(fr.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
    except Exception:  # noqa: BLE001
        fr = np.zeros_like(gray)
    deep = depth > 14
    root = (deep & ((fr > 0.22) | (sat >= 0.12))).astype(np.uint8)
    root = cv2.morphologyEx(root, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    shadow = (deep & (sat < 0.12) & ~(fr > 0.22)).astype(np.uint8)
    shadow = cv2.morphologyEx(shadow, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    return rgb, gray, sat, depth, root, shadow


def crop_at(rgb, mask_root, mask_shadow, cy, cx, side=CROP):
    h, w = rgb.shape[:2]
    y0 = int(np.clip(cy - side // 2, 0, max(0, h - side)))
    x0 = int(np.clip(cx - side // 2, 0, max(0, w - side)))
    sl = (slice(y0, y0 + side), slice(x0, x0 + side))
    return rgb[sl], mask_root[sl], mask_shadow[sl]


def render(rgb_c, root_c, sh_c, title, note, lines):
    ov = rgb_c.astype(np.float32).copy()
    ov[root_c] = ov[root_c] * 0.35 + np.array([0, 200, 0]) * 0.65
    ov[sh_c] = ov[sh_c] * 0.45 + np.array([255, 40, 40]) * 0.55
    ov = ov.astype(np.uint8)
    s = 3
    ov = cv2.resize(ov, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
    h, w = ov.shape[:2]
    canvas = Image.new("RGB", (w + 24, h + 120), (255, 255, 255))
    canvas.paste(Image.fromarray(ov), (12, 70))
    dr = ImageDraw.Draw(canvas)
    dr.text((12, 8), title, fill=(0, 0, 0), font=font(26))
    dr.text((12, 40), note, fill=(150, 0, 0), font=font(20))
    y = h + 78
    for ln in lines:
        dr.text((12, y), ln, fill=(0, 0, 0), font=font(19))
        y += 26
    return canvas


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(CFG.read_text(encoding="utf-8"))
    # 两遍法：先跨所有图收集每条规则的全部候选，再做"每张图只出一张示例"的指派，
    # 避免 4 张示例图全落在同一张原图上。
    pool: dict[str, list[dict]] = {}

    def consider(rule, score, it, crop, payload):
        pool.setdefault(rule, []).append(
            {"score": score, "it": it, "crop": crop, "payload": payload})

    for it in cfg["items"]:
        p = PROJ / it["processed_path"]
        rgb, gray, sat, depth, root, shadow = layers(p)
        disp_w = rgb.shape[1]
        f2 = float(it["out_w"]) / disp_w

        dist = cv2.distanceTransform((~root).astype(np.uint8), cv2.DIST_L2, 5)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))
        kn = k / k.sum()

        # 只在中部区域找样例：贴边的目标会被裁剪框夹住，示例图会跑偏
        hh_, ww_ = rgb.shape[:2]
        m0 = CROP // 2 + 20
        interior = np.zeros((hh_, ww_), bool)
        interior[m0:hh_ - m0, m0:ww_ - m0] = True

        for tag, m in [("A", shadow & (dist > 2) & (dist <= 6) & interior),
                       ("B", shadow & (dist > 50) & interior),
                       ("C", shadow & (dist > 25) & interior)]:
            if m.sum() < 200:
                continue
            dens = cv2.filter2D(m.astype(np.float32), -1, kn)
            cy, cx = np.unravel_index(int(np.argmax(dens)), dens.shape)
            score = float(dens[cy, cx])
            consider(tag, score, it, crop_at(rgb, root, shadow, cy, cx),
                     {"pos": [int(cx), int(cy)],
                      "dist_mean_px": float(dist[m].mean()),
                      "sat_mean": float(sat[m].mean()),
                      "f2": f2, "n_px": int(m.sum())})

        n, lab, stats, cent = cv2.connectedComponentsWithStats(root.astype(np.uint8), 8)
        bscore, bcand = 0.0, None
        for i in range(1, n):
            x, y, ww, hh, area = stats[i]
            if area < 40 or area > 600:
                continue
            cx_, cy_ = x + ww // 2, y + hh // 2
            if not (m0 < cx_ < ww_ - m0 and m0 < cy_ < hh_ - m0):
                continue
            elong = max(ww, hh) / max(1, min(ww, hh))
            if elong < 4:
                continue
            s = elong * area ** 0.5
            if s > bscore:
                bscore, bcand = s, (cx_, cy_, ww, hh, elong)
        if bcand:
            x, y, ww, hh, elong = bcand
            consider("D", bscore, it, crop_at(rgb, root, shadow, y, x),
                     {"pos": [int(x), int(y)], "w": int(ww), "h": int(hh),
                      "elong": float(elong), "f2": f2})

    # 指派：每条规则挑"尚未被其它规则占用的图"里证据最强的那张
    best: dict[str, dict] = {}
    used_images: set[str] = set()
    for tag in ["A", "B", "C", "D"]:
        cands = sorted(pool.get(tag, []), key=lambda c: -c["score"])
        free = [c for c in cands if c["it"]["rel_path"] not in used_images]
        chosen = (free or cands)[0] if (free or cands) else None
        if chosen:
            best[tag] = chosen
            used_images.add(chosen["it"]["rel_path"])

    examples = []
    for tag in ["A", "B", "C", "D"]:
        if tag not in best:
            continue
        b = best[tag]
        it, (rgb_c, r_c, s_c), pl = b["it"], b["crop"], b["payload"]
        f2 = pl["f2"]
        if tag == "A":
            title = "规则 A · 贴根半影  ->  判「根」"
            note = "绿=建议判「根」（含紧贴根的柔和暗边）｜红=建议判「阴影」"
            lines = [
                f"实测：该处阴影像素到根缘平均 {pl['dist_mean_px']:.1f}px "
                f"= {pl['dist_mean_px']*MM_PER_PX*f2:.3f} mm（600 dpi），共 {pl['n_px']}px",
                "标准 §1.1(2)：宽度 ≤6px 的贴根暗边计入根类，避免根径被削细、根长被截短",
                f"来源：{it['rel_path']}",
            ]
        elif tag == "B":
            title = "规则 B · 暗但没有线性结构  ->  判「阴影」"
            note = "红=建议判「阴影」｜绿=建议判「根」"
            lines = [
                f"实测：该暗区离最近根平均 {pl['dist_mean_px']:.1f}px "
                f"= {pl['dist_mean_px']*MM_PER_PX*f2:.2f} mm，无管状走向，共 {pl['n_px']}px",
                "标准 §1.2(5)：暗但无线性结构 -> 阴影（本标准的首要判据）",
                f"来源：{it['rel_path']}",
            ]
        elif tag == "C":
            title = "规则 C · 水渍 / 湿痕  ->  判「阴影」"
            note = "红=建议判「阴影」｜绿=建议判「根」"
            lines = [
                f"实测：该区平均饱和度 {pl['sat_mean']:.3f}（低饱和=灰蓝调，与褐色根可分）",
                "标准 §1.2(3)：水渍成块、边缘柔、常呈环状 -> 阴影，绝不能当根",
                f"来源：{it['rel_path']}",
            ]
        else:
            width_px = min(pl["w"], pl["h"]) * f2
            title = "规则 D · 极细根 vs 孤立碎屑  ->  有走向判「根」"
            note = "绿=建议判「根」（有连续走向）｜红=建议判「阴影」"
            lines = [
                f"实测：该细根粗约 {width_px:.0f}px = {width_px*MM_PER_PX:.3f} mm，"
                f"长宽比 {pl['elong']:.1f}",
                "标准 §1.3(1)：能连成走向的极细根判「根」；无走向的孤立暗点判「背景」",
                f"来源：{it['rel_path']}",
            ]
        canvas = render(rgb_c, r_c, s_c, title, note, lines)
        canvas.save(OUT / f"rule_{tag}.png", optimize=True)
        examples.append({"rule": tag, "image": it["rel_path"], "pos_disp": pl["pos"],
                         "score": round(b["score"], 5)})
        print(f"  [{tag}] 来源 {it['rel_path']}  (证据强度 {b['score']:.4f})")

    (OUT / "examples.json").write_text(
        json.dumps({"mm_per_px": MM_PER_PX, "examples": examples}, indent=2,
                   ensure_ascii=False), encoding="utf-8")
    print(f"\n  生成 {len(examples)}/4 张示例图 -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
