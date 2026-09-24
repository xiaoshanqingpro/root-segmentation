"""检查 9.9改mac 里还剩多少贴边残留，并给出更严的清理规则。

上一版规则: 贴边 且 >=80% 像素落在最外圈 3% 带内 -> 删。
漏网原因: 若贴边块有一部分伸进画幅内侧（<80% 在带内），就躲过了规则。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

Image.MAX_IMAGE_PIXELS = None
MAC = Path(r"C:\Users\LIHAOYANG\Desktop\测试\机器\9.9改mac")
BAND = 0.03

for f in sorted(MAC.glob("*-mac.tif")):
    a = np.asarray(Image.open(f).convert("L"))
    fg = a < 128
    H, W = fg.shape
    bh, bw = int(H * BAND), int(W * BAND)
    band = np.zeros_like(fg)
    band[:bh, :] = band[-bh:, :] = band[:, :bw] = band[:, -bw:] = True
    n, lab, st, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), 8)
    tot_band = int((fg & band).sum())
    rows = []
    for j in range(1, n):
        x, y, w, h, area = st[j]
        touch = (x <= 0 or y <= 0 or x + w >= W or y + h >= H)
        if not touch:
            continue
        m = lab == j
        bf = float(band[m].mean())
        rows.append((area, bf, w, h))
    rows.sort(reverse=True)
    print(f"{f.stem:<14} 贴边带内前景 {tot_band:>7}px   贴边连通块 {len(rows)} 个")
    for area, bf, w, h in rows[:4]:
        print(f"      面积 {area:>7}  带内占比 {bf:5.2f}  外接框 {w}x{h}")
