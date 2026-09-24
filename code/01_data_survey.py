"""阶段一 · 数据勘察：逐张记录尺寸 / 格式 / DPI / 色彩分布 / 阴影代理指标。

产出:
  outputs/阶段一/survey.csv      逐张记录（后续测量与 dpi 换算要用）
  outputs/阶段一/survey_summary.json

设计要点:
  - 用 PIL draft 让 JPEG 以 1/2,1/4,1/8 解码，避免 600dpi 大图全解码拖慢速度
  - 所有统计在"长边 900px"的缩略图上做，比例指标与分辨率无关
  - 阴影代理: 背景(V 高分位)之下、且低饱和 = 阴影嫌疑; 背景之下、且带褐色饱和 = 根嫌疑
    （这只是给勘察用的粗略代理，真正的判据由阶段一人工确认后写进标注标准）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# --- 统一路径（见 paths.py；数据搬家只改那里）---
sys.path.insert(0, str(Path(__file__).parent))
import paths as P  # noqa: E402

IMAGE_DIR = P.RAW_D1
OUT_DIR = Path(r"D:\根系分割项目\outputs\阶段一")
LONG_SIDE = 900


def otsu(gray: np.ndarray) -> int:
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    idx = np.arange(256)
    sum_all = (idx * hist).sum()
    sum_b, w_b = 0.0, 0.0
    best, thr = -1.0, 128
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f <= 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        var = w_b * w_f * (m_b - m_f) ** 2
        if var > best:
            best, thr = var, t
    return thr


def load_small(path: Path, long_side: int = LONG_SIDE) -> Image.Image:
    im = Image.open(path)
    im.draft("RGB", (long_side, long_side))  # JPEG 快速降采样解码
    im = im.convert("RGB")
    scale = long_side / max(im.size)
    if scale < 1:
        im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))),
                       Image.BILINEAR)
    return im


def rgb_to_hsv_np(rgb: np.ndarray) -> np.ndarray:
    arr = rgb.astype(np.float32) / 255.0
    mx = arr.max(axis=2)
    mn = arr.min(axis=2)
    diff = mx - mn
    sat = np.where(mx > 0, diff / np.maximum(mx, 1e-6), 0.0)
    return np.stack([mx, sat], axis=2)  # V, S 足够我们用


def analyze(path: Path) -> dict:
    rec: dict = {}
    with Image.open(path) as raw:
        rec["width"], rec["height"] = raw.size
        rec["pil_mode"] = raw.mode
        rec["format"] = raw.format
        rec["jpeg_progressive"] = bool(raw.info.get("progressive", False))
        dpi = raw.info.get("dpi")
        rec["dpi_x"], rec["dpi_y"] = (round(float(dpi[0]), 2), round(float(dpi[1]), 2)) if dpi else (None, None)
        rec["icc_profile"] = bool(raw.info.get("icc_profile"))
        rec["exif_present"] = bool(raw.getexif())

    rec["file_bytes"] = path.stat().st_size
    rec["mtime"] = path.stat().st_mtime

    im = load_small(path)
    rec["proc_w"], rec["proc_h"] = im.size
    arr = np.asarray(im)
    v, s = rgb_to_hsv_np(arr)[..., 0], rgb_to_hsv_np(arr)[..., 1]

    gray = np.asarray(im.convert("L"))
    thr = otsu(gray)
    rec["otsu_thr"] = int(thr)
    rec["dark_pct"] = round(float((gray < thr).mean() * 100), 3)

    # 背景亮度: 取 V 的 95 分位（大面积白托盘）
    bg_v = float(np.percentile(v, 95))
    rec["v_mean"] = round(float(v.mean()), 4)
    rec["v_bg"] = round(bg_v, 4)
    rec["v_p05"] = round(float(np.percentile(v, 5)), 4)
    rec["s_mean"] = round(float(s.mean()), 4)

    darker = v < (bg_v - 0.06)          # 比背景明显暗
    low_sat = s < 0.10                  # 灰暗无彩 = 阴影/污渍嫌疑
    brown = s >= 0.10                   # 带彩 = 根/杂物嫌疑
    rec["pixel_darker_pct"] = round(float(darker.mean() * 100), 3)
    rec["proxy_shadow_pct"] = round(float((darker & low_sat).mean() * 100), 3)
    rec["proxy_root_pct"] = round(float((darker & brown).mean() * 100), 3)

    # 背景不均匀度: 把背景像素(V 高于阈) 分 6x6 网格，看各格背景均值极差
    bgmask = ~darker
    h, w = v.shape
    grid = []
    for i in range(6):
        for j in range(6):
            blk = v[i * h // 6:(i + 1) * h // 6, j * w // 6:(j + 1) * w // 6]
            bm = bgmask[i * h // 6:(i + 1) * h // 6, j * w // 6:(j + 1) * w // 6]
            grid.append(float(blk[bm].mean()) if bm.any() else np.nan)
    grid = np.array(grid)
    rec["bg_grid_min"] = round(float(np.nanmin(grid)), 4)
    rec["bg_grid_max"] = round(float(np.nanmax(grid)), 4)
    rec["bg_grid_range"] = round(float(np.nanmax(grid) - np.nanmin(grid)), 4)

    # 外圈 2% 边框的暗像素比例 -> 托盘边缘 / 裁切残留
    m = max(2, int(min(h, w) * 0.02))
    ring = np.zeros_like(gray, dtype=bool)
    ring[:m, :] = ring[-m:, :] = ring[:, :m] = ring[:, -m:] = True
    rec["border_dark_pct"] = round(float((gray[ring] < thr).mean() * 100), 3)
    return rec


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(IMAGE_DIR.rglob("*.jpg")) + sorted(IMAGE_DIR.rglob("*.JPG"))
    files = sorted(set(files))
    print(f"[survey] 发现 {len(files)} 张 jpg", flush=True)

    rows = []
    for i, p in enumerate(files, 1):
        try:
            rec = analyze(p)
            rec["rel_path"] = str(p.relative_to(IMAGE_DIR.parent))
            rec["batch"] = p.parent.name
            rows.append(rec)
            print(f"[{i:3d}/{len(files)}] OK  {p.name}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[{i:3d}/{len(files)}] FAIL {p.name}: {type(exc).__name__}: {exc}", flush=True)

    import csv
    keys = ["rel_path", "batch", "width", "height", "dpi_x", "dpi_y", "format", "pil_mode",
            "file_bytes", "jpeg_progressive", "icc_profile", "exif_present",
            "otsu_thr", "dark_pct", "v_mean", "v_bg", "v_p05", "s_mean",
            "pixel_darker_pct", "proxy_shadow_pct", "proxy_root_pct",
            "bg_grid_min", "bg_grid_max", "bg_grid_range", "border_dark_pct", "mtime"]
    with (OUT_DIR / "survey.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        wtr = csv.DictWriter(fh, fieldnames=keys)
        wtr.writeheader()
        for r in rows:
            wtr.writerow({k: r.get(k) for k in keys})

    sizes: dict[str, int] = {}
    dpis: dict[str, int] = {}
    for r in rows:
        sizes[f'{r["width"]}x{r["height"]}'] = sizes.get(f'{r["width"]}x{r["height"]}', 0) + 1
        dpis[f'{r["dpi_x"]}'] = dpis.get(f'{r["dpi_x"]}', 0) + 1
    summary = {
        "n_images": len(rows),
        "size_distribution": sizes,
        "dpi_distribution": dpis,
        "batches": {b: sum(1 for r in rows if r["batch"] == b) for b in sorted({r["batch"] for r in rows})},
    }
    (OUT_DIR / "survey_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
