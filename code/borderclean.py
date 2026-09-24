"""可复用的边框自动清理模块（从 47_auto_clean_borders.py 抽出）。

两遍清理，都只动"贴到图像最外缘的连通块"，绝不用矩形硬切：
  1. 深色黑边：灰度 < DARK_T，且位于逐边扫描出的黑边带内，且连通块确实贴到最外缘
  2. 浅色纸边/盖板边：非白 + 贴最外缘 + 外接框跨越大半个画幅 + 整体偏亮

实测（2026-09-23）：清掉 22-001 / 44-001 / 46-002 的扫描黑边后，
U-Net 整图 Dice 中位从 0.526 升到 0.845。
"""
from __future__ import annotations

import cv2
import numpy as np

DARK_T = 110        # 灰度低于此值算"深色"
COVER = 0.35        # 该行/列深色贯通率超过此值，认为还在黑边带里
MAX_BAND = 500      # 单边最大带宽（防意外）
NONWHITE = 250
LIGHT_SPAN = 0.40   # 浅色边条：外接框至少要跨画幅的这个比例
LIGHT_GRAY_MIN = 140  # 浅色边条：整体均值灰度下限（低于此值可能是根，不动）


def _band_width(prof: np.ndarray) -> int:
    """从边缘往里扫，返回"贯通带"宽度"""
    w = 0
    n = min(MAX_BAND, len(prof))
    for i in range(n):
        if prof[i] > COVER:
            w = i + 1
        elif i > 4 and prof[i] <= COVER and prof[min(i + 3, n - 1)] <= COVER:
            break
    return w


def clean_borders(rgb: np.ndarray, dark_t: int = DARK_T) -> tuple[np.ndarray, dict]:
    """返回 (清理后的 rgb, 统计)。不修改入参。"""
    out = rgb.copy()
    gray = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
    H, W = gray.shape
    dark = gray < dark_t

    top = _band_width(dark.mean(axis=1))
    bot = _band_width(dark.mean(axis=1)[::-1])
    left = _band_width(dark.mean(axis=0))
    right = _band_width(dark.mean(axis=0)[::-1])

    band = np.zeros_like(dark)
    if top:
        band[:top, :] = True
    if bot:
        band[-bot:, :] = True
    if left:
        band[:, :left] = True
    if right:
        band[:, -right:] = True

    # 第一遍：深色黑边（只取贴到最外缘的连通块）
    cand = dark & band
    n, lab, _, _ = cv2.connectedComponentsWithStats(cand.astype(np.uint8), 8)
    on_edge = np.unique(np.concatenate([lab[0, :], lab[H - 1, :],
                                        lab[:, 0], lab[:, W - 1]]))
    keep = np.zeros(n, bool)
    keep[on_edge] = True
    keep[0] = False
    remove = keep[lab]
    n_dark = int(remove.sum())
    out[remove] = 255

    # 第二遍：浅色纸边 / 盖板边
    g2 = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
    nw2 = np.any(out < NONWHITE, axis=2)
    n2, l2, s2, _ = cv2.connectedComponentsWithStats(nw2.astype(np.uint8), 8)
    light = np.zeros_like(nw2)
    for j in range(1, n2):
        x, y, w_, h_ = (s2[j, cv2.CC_STAT_LEFT], s2[j, cv2.CC_STAT_TOP],
                        s2[j, cv2.CC_STAT_WIDTH], s2[j, cv2.CC_STAT_HEIGHT])
        if not (x <= 2 or y <= 2 or x + w_ >= W - 2 or y + h_ >= H - 2):
            continue
        if not ((w_ >= LIGHT_SPAN * W) or (h_ >= LIGHT_SPAN * H)):
            continue
        m = l2 == j
        if float(g2[m].mean()) < LIGHT_GRAY_MIN:
            continue          # 整体偏暗 -> 可能是根，不动
        light |= m
    n_light = int(light.sum())
    out[light] = 255

    return out, {"band_top": top, "band_bottom": bot, "band_left": left, "band_right": right,
                 "removed_dark_px": n_dark, "removed_light_px": n_light,
                 "removed_total_px": n_dark + n_light}
