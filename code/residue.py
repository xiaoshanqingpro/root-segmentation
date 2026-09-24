"""残留清理（机器图后处理的统一实现）。

从 `55_make_machine_one.py` 抽出，供批量脚本复用，保证口径一致。

处理三类残留:
  A. 贴边成带的扫描仪盖板/黑边       -> 删
  B. 贴边浅条 / 整个外接框落在最外圈内 -> 删（**不要求接触边界**，这是关键）
  C. 远离主体的游离碎屑、以及面积大但形状是团块的杂物 -> 删
  保留: 贴近主体的碎块（很可能是被管线打断的细根，删了会更欠分割）

输入输出都是**图像**（根=0 黑, 背景=255 白），不是布尔掩膜。
"""
from __future__ import annotations

import cv2
import numpy as np

BIG = 2000              # 面积 >= 此值算"大块"
KEEP_DIST = 80          # 离主体 <= 此距离的碎块保留（断根）
BAND_FRAC = 0.03        # 最外圈窄带
BAND_IN = 0.50          # 贴边块带内占比 >= 此值 -> 判扫描边
EDGE_DEPTH = 0.15       # 贴边块垂直边界尺寸 < 画幅此比例 -> 判沿边浅条
EDGE_MARGIN = 0.20      # 贴边块整个外接框在最外圈内 -> 判扫描边
EDGE_MARGIN_HARD = 0.06 # 不接触边界但整个外接框在最外圈内 -> 也判扫描边
ELONG_ROOT = 3.0        # sqrt(面积)/半宽 达到此值才算"根状"
BLOB_DIST = 30          # 团块离主体超过此距离 -> 判杂物


def clean_residue(img: np.ndarray) -> tuple[np.ndarray, dict]:
    fg = img < 128
    H, W = fg.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), 8)
    band = np.zeros_like(fg)
    bh, bw = int(H * BAND_FRAC), int(W * BAND_FRAC)
    band[:bh, :] = band[-bh:, :] = band[:, :bw] = band[:, -bw:] = True
    drop = np.zeros_like(fg)
    se: dict = {}

    # --- 向量化的逐块统计（关键性能点）---
    # 原来对每个连通块单独算一次 distanceTransform / 单独取 mask，
    # 一张图上千个块 -> 单张 24s。改成**整图算一次 DT + np.maximum.at 聚合**。
    dt_all = cv2.distanceTransform(fg.astype(np.uint8), cv2.DIST_L2, 5)
    flat_lab = lab.ravel()
    max_dt = np.zeros(n, dtype=np.float32)
    np.maximum.at(max_dt, flat_lab, dt_all.ravel())
    # 每个块的带内像素数（一次 bincount 搞定）
    band_cnt = np.bincount(flat_lab[band.ravel()], minlength=n)
    area_all = np.bincount(flat_lab, minlength=n)

    elong = {j: (float(st[j, cv2.CC_STAT_AREA]) ** 0.5) / max(0.5, float(max_dt[j]))
             for j in range(1, n)}

    largest = max(range(1, n), key=lambda j: st[j, cv2.CC_STAT_AREA], default=0)
    main_ids = {j for j in range(1, n)
                if j == largest or (st[j, cv2.CC_STAT_AREA] >= BIG and elong[j] >= ELONG_ROOT)}
    main = np.isin(lab, list(main_ids)) if main_ids else np.zeros_like(fg)
    dist = cv2.distanceTransform((~main).astype(np.uint8), cv2.DIST_L2, 5)
    # 每个块到主体的最近距离（同样一次 minimum.at 聚合）
    min_dist = np.full(n, np.inf, dtype=np.float32)
    np.minimum.at(min_dist, flat_lab, dist.ravel())

    drop_ids: set[int] = set()
    for j in range(1, n):
        x, y, w, h, area = st[j]
        t_top, t_bot, t_left, t_right = (y <= 0), (y + h >= H), (x <= 0), (x + w >= W)
        touch = t_top or t_bot or t_left or t_right
        d = float(min_dist[j])
        # 硬余量（放在最前，不受"接触边界"与"主体豁免"限制）
        if (x + w <= EDGE_MARGIN_HARD * W) or (x >= (1 - EDGE_MARGIN_HARD) * W) or \
           (y + h <= EDGE_MARGIN_HARD * H) or (y >= (1 - EDGE_MARGIN_HARD) * H):
            drop_ids.add(j)
            se["edge_margin"] = se.get("edge_margin", 0) + int(area)
            continue
        if touch:
            shallow = ((t_top or t_bot) and h < EDGE_DEPTH * H) or \
                      ((t_left or t_right) and w < EDGE_DEPTH * W)
            in_margin = (x + w <= EDGE_MARGIN * W) or (x >= (1 - EDGE_MARGIN) * W) or \
                        (y + h <= EDGE_MARGIN * H) or (y >= (1 - EDGE_MARGIN) * H)
            bf = float(band_cnt[j]) / max(1, int(area_all[j]))
            if bf >= BAND_IN:
                drop_ids.add(j); se["edge"] = se.get("edge", 0) + int(area); continue
            if shallow or in_margin:
                drop_ids.add(j); se["edge_strip"] = se.get("edge_strip", 0) + int(area); continue
            if d > KEEP_DIST:
                drop_ids.add(j); se["edge_iso"] = se.get("edge_iso", 0) + int(area); continue
        else:
            if j in main_ids:
                continue
            if area < BIG and d > KEEP_DIST:
                drop_ids.add(j); se["debris"] = se.get("debris", 0) + int(area); continue
            if elong[j] < ELONG_ROOT and d > BLOB_DIST:
                drop_ids.add(j); se["blob"] = se.get("blob", 0) + int(area); continue
            se["kept"] = se.get("kept", 0) + int(area)

    drop = np.isin(lab, list(drop_ids)) if drop_ids else np.zeros_like(fg)
    keep = fg & ~drop
    return np.where(keep, 0, 255).astype(np.uint8), se
