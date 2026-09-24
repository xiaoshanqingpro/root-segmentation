"""阶段一 · 抽样可视化：随机抽 8 张，标出"我判定的阴影区域"，供人工校对理解是否一致。

方法（经典、可解释，不用深度学习）:
  1. 大核形态学闭运算估计"局部背景照度" illum  —— 根与阴影都被填平，只留照明不均
  2. depth = illum - gray  → 比局部背景暗多少
  3. 反射率校正 corrected = gray / illum * mean(illum)
  4. Frangi 血管增强（在反相图上找暗色管状结构）→ 有线性结构的才算"根状"
  5. 阴影嫌疑 = 明显变暗 且 低饱和 且 无管状响应
     根   嫌疑 = 明显变暗 且 (有管状响应 或 有褐色饱和)

产出:
  outputs/阶段一/samples/<name>_panel.png   原图 | 阴影嫌疑叠加 | 根嫌疑叠加 | 照度校正对比
  outputs/阶段一/samples/<name>_proposal_shadow.png  阴影嫌疑二值图（白=阴影嫌疑）
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
IMAGE_DIR = P.RAW_D1
OUT_DIR = PROJ / "outputs" / "阶段一"
SAMPLE_DIR = OUT_DIR / "samples"
SEED = 42
N_SAMPLES = 8
WORK_LONG_SIDE = 1100


def load_rgb(path: Path, long_side: int = WORK_LONG_SIDE) -> tuple[np.ndarray, float]:
    im = Image.open(path)
    im.draft("RGB", (long_side, long_side))
    im = im.convert("RGB")
    scale = long_side / max(im.size)
    if scale < 1:
        im = im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS)
    return np.asarray(im), scale


def illumination(gray: np.ndarray, k: int = 61) -> np.ndarray:
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(gray, cv2.MORPH_CLOSE, ker)


def analyze_image(path: Path) -> dict:
    rgb, scale = load_rgb(path)
    h, w = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1].astype(np.float32) / 255.0
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

    illum = illumination(gray.astype(np.uint8)).astype(np.float32)
    depth = illum - gray                       # >0 表示比局部背景暗
    corrected = np.clip(gray / np.maximum(illum, 1e-3) * float(illum.mean()), 0, 255)

    inv = (255.0 - corrected).astype(np.float32)
    frangi = cv2.normalize(
        cv2.GaussianBlur(inv, (0, 0), 1.0), None, 0, 1, cv2.NORM_MINMAX
    )
    try:
        from skimage.filters import frangi as sk_frangi
        fr = sk_frangi(inv / 255.0, sigmas=range(1, 7), black_ridges=False)
        frangi = cv2.normalize(fr.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] skimage.frangi 不可用, 用形态学替代: {exc}", flush=True)
        bh = cv2.morphologyEx(corrected.astype(np.uint8), cv2.MORPH_BLACKHAT,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        frangi = cv2.normalize(bh.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)

    deep = depth > 14.0
    low_sat = sat < 0.12
    tubular = frangi > 0.22
    shadow = deep & low_sat & ~tubular
    root = deep & (tubular | (sat >= 0.12))

    # 清理: 去掉过小的孤立斑点
    shadow_u8 = cv2.morphologyEx(shadow.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    root_u8 = cv2.morphologyEx(root.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    return {
        "rgb": rgb, "gray": gray, "illum": illum, "depth": depth, "corrected": corrected,
        "frangi": frangi, "sat": sat,
        "shadow": shadow_u8.astype(bool), "root": root_u8.astype(bool), "scale": scale,
    }


def overlay(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.55) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    m = mask.astype(bool)
    for c in range(3):
        out[..., c][m] = out[..., c][m] * (1 - alpha) + color[c] * alpha
    return out.astype(np.uint8)


def label_bar(width: int, texts: list[str], font_size: int = 26) -> np.ndarray:
    bar = np.full((44, width, 3), 255, np.uint8)
    im = Image.fromarray(bar)
    dr = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("msyh.ttc", font_size)
    except Exception:  # noqa: BLE001
        font = ImageFont.load_default()
    seg = width // len(texts)
    for i, t in enumerate(texts):
        dr.text((i * seg + 10, 8), t, fill=(0, 0, 0), font=font)
    return np.asarray(im)


def make_panel(path: Path, res: dict) -> None:
    rgb = res["rgb"]
    h, w = rgb.shape[:2]

    panel = [
        rgb,
        overlay(rgb, res["shadow"], (255, 0, 0)),
        overlay(rgb, res["root"], (0, 170, 0)),
        cv2.cvtColor(res["corrected"].astype(np.uint8), cv2.COLOR_GRAY2RGB),
    ]
    bar = label_bar(w * 2, ["原图", "阴影嫌疑(红)", "根嫌疑(绿)", "照度校正后"])
    grid = np.vstack([bar, np.hstack(panel[:2]), np.hstack(panel[2:])])

    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(SAMPLE_DIR / f"{path.stem}_panel.png")
    Image.fromarray((res["shadow"] * 255).astype(np.uint8)).save(
        SAMPLE_DIR / f"{path.stem}_proposal_shadow.png")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(IMAGE_DIR.rglob("*.jpg"))

    # 分层抽样: 每个批次按比例抽，保证 600dpi / 720dpi 都被覆盖
    rng = np.random.default_rng(SEED)
    by_size: dict[tuple[int, int], list[Path]] = {}
    for p in files:
        with Image.open(p) as im:
            by_size.setdefault(im.size, []).append(p)

    picked: list[Path] = []
    sizes_sorted = sorted(by_size, key=lambda s: -len(by_size[s]))
    for size in sizes_sorted:
        quota = max(1, round(N_SAMPLES * len(by_size[size]) / len(files)))
        quota = min(quota, len(by_size[size]))
        idx = rng.choice(len(by_size[size]), size=quota, replace=False)
        picked += [by_size[size][i] for i in idx]
    picked = sorted(set(picked), key=lambda p: p.name)[:N_SAMPLES + 2]

    print(f"[samples] 尺寸分组: { {f'{k[0]}x{k[1]}': len(v) for k, v in by_size.items()} }", flush=True)
    print(f"[samples] 抽中 {len(picked)} 张: {[p.name for p in picked]}", flush=True)

    stats = []
    for i, p in enumerate(picked, 1):
        print(f"[{i}/{len(picked)}] 处理 {p.name}", flush=True)
        res = analyze_image(p)
        make_panel(p, res)
        stats.append({
            "file": p.name,
            "rel_path": str(p.relative_to(IMAGE_DIR.parent)),
            "shadow_pct": round(float(res["shadow"].mean() * 100), 3),
            "root_pct": round(float(res["root"].mean() * 100), 3),
            "illum_range": round(float(res["illum"].max() - res["illum"].min()), 2),
            "depth_p95": round(float(np.percentile(res["depth"], 95)), 2),
        })
        print(f"    阴影嫌疑 {stats[-1]['shadow_pct']}%  根嫌疑 {stats[-1]['root_pct']}%  "
              f"照度落差 {stats[-1]['illum_range']}", flush=True)

    (OUT_DIR / "samples_stats.json").write_text(
        json.dumps({"seed": SEED, "n": len(picked), "items": stats}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
