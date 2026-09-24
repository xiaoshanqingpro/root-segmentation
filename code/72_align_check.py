"""诊断：人工图(man) 与机器图(CV/UNet) 是否逐像素对齐。"""
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


def load600(p: Path) -> Image.Image:
    im = Image.open(p).convert("RGB")
    d = im.info.get("dpi")
    dpi = float(d[0]) if d else 600.0
    if abs(dpi - 600.0) > 1:
        k = 600.0 / dpi
        im = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), Image.LANCZOS)
    return im


def mask_of(p: Path, thr: int = THR) -> np.ndarray:
    return np.asarray(load600(p).convert("L")) < thr


def bbox_c(m):
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return (0, 0, 0, 0), (0.0, 0.0)
    return (xs.min(), ys.min(), xs.max(), ys.max()), (float(xs.mean()), float(ys.mean()))


def iou(a, b):
    u = int(np.logical_or(a, b).sum())
    return int(np.logical_and(a, b).sum()) / u if u else 0.0


def main() -> int:
    rows = {r["序号"]: r for r in csv.DictReader((B / "_抽样清单.csv").open(encoding="utf-8-sig"))}
    for idx, stem in [("4", "74-003"), ("5", "270-001"), ("1", "11-QG-003")]:
        man = mask_of(B / "人工" / f"{idx}_{stem}-man.jpg")
        cv = mask_of(Path(rows[idx]["CV"]))
        un = mask_of(Path(rows[idx]["UNet"]))
        print(f"=== {idx} {stem}   shape={man.shape} ===")
        for tag, m in [("man", man), ("CV", cv), ("UNet", un)]:
            bb, c = bbox_c(m)
            print(f"  {tag:<5} 根像素 {int(m.sum()):>9}  "
                  f"bbox x[{bb[0]:>5},{bb[2]:>5}] y[{bb[1]:>5},{bb[3]:>5}]  质心 ({c[0]:7.1f},{c[1]:7.1f})")
        raw_cv, raw_un = iou(man, cv), iou(man, un)
        # 平移扫描
        best = (0, 0, 0.0)
        for dy in range(-80, 81, 20):
            for dx in range(-80, 81, 20):
                s = np.roll(np.roll(cv, dy, 0), dx, 1)
                i = iou(man, s)
                if i > best[2]:
                    best = (dx, dy, i)
        print(f"  IoU(man,CV)  = {raw_cv:.4f}     IoU(man,UNet) = {raw_un:.4f}")
        print(f"  平移扫描最优: dx={best[0]:+d} dy={best[1]:+d} -> IoU {best[2]:.4f}")
        # 若平移后仍低，说明不是简单错位
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
