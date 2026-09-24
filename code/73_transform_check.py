"""诊断 man 与机器图之间的几何变换（平移 / 缩放 / 旋转）。"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

Image.MAX_IMAGE_PIXELS = None
B = Path(r"D:\根系分割项目\outputs\对比十张")
THR = 250
DS = 8          # 降采样倍数，加速搜索


def load600(p: Path) -> Image.Image:
    im = Image.open(p).convert("RGB")
    d = im.info.get("dpi")
    dpi = float(d[0]) if d else 600.0
    if abs(dpi - 600.0) > 1:
        k = 600.0 / dpi
        im = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), Image.LANCZOS)
    return im


def mask_of(p: Path) -> np.ndarray:
    return np.asarray(load600(p).convert("L")) < THR


def moments(m: np.ndarray) -> tuple:
    ys, xs = np.nonzero(m)
    return (xs.mean(), ys.mean(), xs.std(), ys.std())


def iou(a, b):
    u = int(np.logical_or(a, b).sum())
    return int(np.logical_and(a, b).sum()) / u if u else 0.0


def main() -> int:
    rows = {r["序号"]: r for r in csv.DictReader((B / "_抽样清单.csv").open(encoding="utf-8-sig"))}
    for idx, stem in [("5", "270-001"), ("4", "74-003"), ("1", "11-QG-003")]:
        man = mask_of(B / "人工" / f"{idx}_{stem}-man.jpg")
        cv = mask_of(Path(rows[idx]["CV"]))
        mx, my, msx, msy = moments(man)
        cx, cy, csx, csy = moments(cv)
        print(f"=== {idx} {stem} ===")
        print(f"  man 质心({mx:7.1f},{my:7.1f}) 标准差({msx:6.1f},{msy:6.1f})  "
              f"有效范围 x±{3*msx:.0f} y±{3*msy:.0f}")
        print(f"  CV  质心({cx:7.1f},{cy:7.1f}) 标准差({csx:6.1f},{csy:6.1f})")
        print(f"  **尺度比 man/CV = {msx/csx:.5f} (x)  {msy/csy:.5f} (y)**")
        print(f"  质心位移 = ({mx-cx:+.1f}, {my-cy:+.1f}) px")

        # 降采样后做 缩放+平移 搜索
        sm = np.asarray(Image.fromarray(man.astype(np.uint8) * 255)
                        .resize((man.shape[1] // DS, man.shape[0] // DS), Image.BILINEAR)) > 127
        sc = np.asarray(Image.fromarray(cv.astype(np.uint8) * 255)
                        .resize((cv.shape[1] // DS, cv.shape[0] // DS), Image.BILINEAR)) > 127
        h, w = sc.shape
        base = iou(sm, sc)
        best = (1.0, 0, 0, base)
        for k in np.arange(0.96, 1.045, 0.005):
            for dy in range(-30, 31, 3):
                for dx in range(-30, 31, 3):
                    t = np.asarray(Image.fromarray(sc.astype(np.uint8) * 255).resize(
                        (max(1, int(w * k)), max(1, int(h * k))), Image.BILINEAR)) > 127
                    if t.shape != sm.shape:
                        c = np.zeros_like(sm)
                        hh = min(sm.shape[0], t.shape[0]); ww = min(sm.shape[1], t.shape[1])
                        c[:hh, :ww] = t[:hh, :ww]; t = c
                    s = np.roll(np.roll(t, dy, 0), dx, 1)
                    v = iou(sm, s)
                    if v > best[3]:
                        best = (k, dx * DS, dy * DS, v)
        print(f"  降采样 IoU(未变换) = {base:.4f}")
        print(f"  **最佳: 缩放 {best[0]:.3f}  平移 ({best[1]:+d},{best[2]:+d})px  -> IoU {best[3]:.4f}**\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
