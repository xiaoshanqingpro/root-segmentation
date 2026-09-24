"""诊断 G218-001 漏掉的细根到底丢在哪一步。

候选损失点:
  A. 洪水填充(容差20) 把细根当背景填掉了   -> 它压根不在 fg 里
  B. 在 fg 里，但被"够暗/管状/饱和"判据滤掉
  C. 在 core 里，但被连通块过滤/收边/平滑去掉
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import rootseg_common as R  # noqa: E402

# --- 统一路径（见 paths.py；数据搬家只改那里）---
sys.path.insert(0, str(Path(__file__).parent))
import paths as P  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
MAN = P.TEST_MAN
ORIG = P.TEST_ORIG
THR = 210


def orig_of(stem):
    for e in (".tif", ".jpg", ".png"):
        p = ORIG / (stem + e)
        if p.exists():
            return p


def main() -> int:
    for s in ["G218-001", "134-003", "7-003", "66-003"]:
        pm, po = MAN / f"{s}-man.tif", orig_of(s)
        if not pm.exists() or po is None:
            continue
        man = np.asarray(Image.open(pm).convert("L")) < THR
        rgb = np.asarray(Image.open(po).convert("RGB"))
        H, W = rgb.shape[:2]

        # 复现中间步骤（特征尺度）
        sc = min(1.0, R.FEAT_LONG / max(H, W))
        small = cv2.resize(rgb, (int(W * sc), int(H * sc)), interpolation=cv2.INTER_AREA)
        hh, ww = small.shape[:2]
        bg = R.flood_background(small)
        fg = ~bg

        # 把人工掩膜降到特征尺度
        man_s = cv2.resize(man.astype(np.uint8), (ww, hh), interpolation=cv2.INTER_NEAREST).astype(bool)

        hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
        sat = hsv[..., 1].astype(np.float32) / 255.0
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32)
        k = max(31, (int(min(hh, ww) * 0.05) // 2) * 2 + 1)
        illum = cv2.morphologyEx(gray.astype(np.uint8), cv2.MORPH_CLOSE,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(np.float32)
        depth = illum - gray

        tot = int(man_s.sum())
        in_fg = int((man_s & fg).sum())
        deep = depth > R.DARK_DROP
        in_deep = int((man_s & fg & deep).sum())
        core = fg & deep & ((sat >= R.SAT_ROOT) | ((gray < R.GRAY_TUB)))
        in_core = int((man_s & core).sum())
        print(f"\n{s}:  人工掩膜(特征尺度) {tot} px")
        print(f"  在洪水填充前景 fg 内 : {in_fg:>8} ({100*in_fg/max(1,tot):5.1f}%)   "
              f"<- 丢失 {tot-in_fg} px 是洪水填充造成的")
        print(f"  且够暗 depth>{R.DARK_DROP}     : {in_deep:>8} ({100*in_deep/max(1,tot):5.1f}%)")
        print(f"  且过核心判据 core    : {in_core:>8} ({100*in_core/max(1,tot):5.1f}%)")
        if tot - in_fg > 0:
            miss = man_s & ~fg
            print(f"  被洪水填充吞掉的像素: 灰度中位 {np.median(gray[miss]):.0f}, "
                  f"暗度中位 {np.median(depth[miss]):.1f}  "
                  f"(容差 {R.FLOOD_TOL})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
