"""阶段一 · 经典方法基线：照度校正 + bottom-hat / Frangi 血管增强。

目的有两个:
  A. 校准"深度方案是否必要"——看经典方法在阴影干扰下能到什么程度
  B. 作为后续深度模型的对照基线

对阶段一抽样的每张图，各跑 5 条流水线并量化对比:
  L1 直接 Otsu（不校正，最容易把阴影算成根）
  L2 照度校正 + Otsu
  L3 bottom-hat + Otsu
  L4 Frangi 管状增强 + 阈值
  L5 校正 + bottom-hat 取交集，并用"低饱和"剔除阴影

指标口径:
  rootarea_pct   被判定为根的像素比例
  shadow_fp_pct  阴影嫌疑区内被判成根的像素 / 阴影嫌疑区（阴影误判为根的比例，越高越糟）
  recall_proxy   在"根嫌疑区"内的召回（用勘察代理当你的人眼标注，只作相对比较）

产出: outputs/阶段一/baseline/<name>_baseline.png , baseline_metrics.csv
"""
from __future__ import annotations

import csv
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
IMAGE_DIR = P.RAW_D1
OUT_DIR = PROJ / "outputs" / "阶段一" / "baseline"
SAMPLE_STATS = PROJ / "outputs" / "阶段一" / "samples_stats.json"
WORK_LONG_SIDE = 1100


def load_rgb(path: Path, long_side: int = WORK_LONG_SIDE) -> np.ndarray:
    im = Image.open(path)
    im.draft("RGB", (long_side, long_side))
    im = im.convert("RGB")
    scale = long_side / max(im.size)
    if scale < 1:
        im = im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS)
    return np.asarray(im)


def otsu_mask(img: np.ndarray) -> np.ndarray:
    u8 = np.clip(img, 0, 255).astype(np.uint8)
    _, m = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return m > 0


def illumination(gray: np.ndarray, k: int = 61) -> np.ndarray:
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(gray.astype(np.uint8), cv2.MORPH_CLOSE, ker).astype(np.float32)


def frangi_map(gray: np.ndarray) -> np.ndarray:
    inv = 255.0 - gray
    try:
        from skimage.filters import frangi as sk_frangi
        fr = sk_frangi(inv / 255.0, sigmas=range(1, 7), black_ridges=False)
        return cv2.normalize(fr.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
    except Exception:  # noqa: BLE001
        bh = cv2.morphologyEx(gray.astype(np.uint8), cv2.MORPH_BLACKHAT,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        return cv2.normalize(bh.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)


def build_layers(rgb: np.ndarray) -> dict:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1].astype(np.float32) / 255.0
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    illum = illumination(gray)
    corrected = np.clip(gray / np.maximum(illum, 1e-3) * float(illum.mean()), 0, 255)
    bh = cv2.morphologyEx(corrected.astype(np.uint8), cv2.MORPH_BLACKHAT,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))).astype(np.float32)
    fr = frangi_map(corrected)

    depth = illum - gray
    shadow_proxy = (depth > 14) & (sat < 0.12) & (fr < 0.22)
    root_proxy = (depth > 14) & (~shadow_proxy)

    layers = {
        "gray": gray, "illum": illum, "corrected": corrected, "blackhat": bh,
        "frangi": fr, "sat": sat,
        "L1": otsu_mask(gray),
        "L2": otsu_mask(corrected),
        "L3": otsu_mask(cv2.normalize(bh, None, 0, 255, cv2.NORM_MINMAX)),
        "L4": fr > 0.25,
        "L5": None,
        "shadow_proxy": shadow_proxy,
        "root_proxy": root_proxy,
    }
    l5 = otsu_mask(cv2.normalize(bh, None, 0, 255, cv2.NORM_MINMAX)) & (sat >= 0.10)
    l5 = cv2.morphologyEx(l5.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    layers["L5"] = l5
    return layers


def overlay(rgb: np.ndarray, mask: np.ndarray, color, alpha: float = 0.5) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    m = mask.astype(bool)
    for c in range(3):
        out[..., c][m] = out[..., c][m] * (1 - alpha) + color[c] * alpha
    return out.astype(np.uint8)


def label_bar(width: int, texts: list[str], font_size: int = 22) -> np.ndarray:
    im = Image.fromarray(np.full((40, width, 3), 255, np.uint8))
    dr = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("msyh.ttc", font_size)
    except Exception:  # noqa: BLE001
        font = ImageFont.load_default()
    seg = width // len(texts)
    for i, t in enumerate(texts):
        dr.text((i * seg + 8, 8), t, fill=(0, 0, 0), font=font)
    return np.asarray(im)


def make_baseline_figure(path: Path, rgb: np.ndarray, layers: dict) -> None:
    h, w = rgb.shape[:2]
    cols = [
        rgb,
        overlay(rgb, layers["L1"], (255, 0, 0)),
        overlay(rgb, layers["L2"], (255, 0, 0)),
        overlay(rgb, layers["L3"], (255, 0, 0)),
        overlay(rgb, layers["L4"], (255, 0, 0)),
        overlay(rgb, layers["L5"], (0, 170, 0)),
    ]
    titles = ["原图", "L1 直接Otsu", "L2 照度校正+Otsu", "L3 bottom-hat",
              "L4 Frangi", "L5 校正+bh+饱和度约束"]
    bar = label_bar(w * 3, titles)
    row1 = np.hstack(cols[:3])
    row2 = np.hstack(cols[3:])
    grid = np.vstack([bar, row1, row2])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(OUT_DIR / f"{path.stem}_baseline.png")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = json.loads(SAMPLE_STATS.read_text(encoding="utf-8"))
    rows = []
    for i, item in enumerate(meta["items"], 1):
        p = Path(r"C:\Users\LIHAOYANG\Desktop\扫描") / item["rel_path"]
        print(f"[{i}/{len(meta['items'])}] 基线 {p.name}", flush=True)
        rgb = load_rgb(p)
        L = build_layers(rgb)
        make_baseline_figure(p, rgb, L)

        sh = L["shadow_proxy"]
        rt = L["root_proxy"]
        sh_area = max(int(sh.sum()), 1)
        rt_area = max(int(rt.sum()), 1)
        rec = {"file": p.name, "rel_path": item["rel_path"],
               "shadow_proxy_pct": round(float(sh.mean() * 100), 3),
               "root_proxy_pct": round(float(rt.mean() * 100), 3)}
        for name in ["L1", "L2", "L3", "L4", "L5"]:
            m = L[name]
            rec[f"{name}_pct"] = round(float(m.mean() * 100), 3)
            # 阴影误判为根: 阴影嫌疑区里被这条流水线判成根的比例
            rec[f"{name}_shadowFP_pct"] = round(float((m & sh).sum() / sh_area * 100), 2)
            rec[f"{name}_rootRecall_pct"] = round(float((m & rt).sum() / rt_area * 100), 2)
        rows.append(rec)

    keys = list(rows[0].keys())
    with (OUT_DIR / "baseline_metrics.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)

    print("\n=== 经典流水线平均表现（阴影误判为根 越低越好 / 根召回 越高越好）===")
    print(f"{'流水线':<6}{'阴影FP%':>10}{'根召回%':>10}{'根面积占比%':>12}")
    for name, label in [("L1", "L1"), ("L2", "L2"), ("L3", "L3"), ("L4", "L4"), ("L5", "L5")]:
        fp = np.mean([r[f"{name}_shadowFP_pct"] for r in rows])
        rc = np.mean([r[f"{name}_rootRecall_pct"] for r in rows])
        ar = np.mean([r[f"{name}_pct"] for r in rows])
        print(f"{label:<6}{fp:>10.2f}{rc:>10.2f}{ar:>12.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
