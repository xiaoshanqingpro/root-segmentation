"""分析机器图里的"残留"：边缘黑块 + 游离碎片，并尝试清除。

用户问: "机器图边缘的黑块和游离的碎片无法清除吗"

两类残留:
  贴边黑块 = 连通块碰到画幅边缘（扫描仪黑边/盖板边没被完全判成背景）
  游离碎片 = 不与主体根系相连的小连通块（纸屑、灰尘、被误判的斑点）
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# --- 统一路径（见 paths.py；数据搬家只改那里）---
sys.path.insert(0, str(Path(__file__).parent))
import paths as P  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

Image.MAX_IMAGE_PIXELS = None
MAC = P.TEST_MAC
OUT = Path(r"D:\根系分割项目\outputs\残留分析")
SAMPLES = ["28-CL-003", "126-003", "134-003", "66-003", "G218-001", "7-003"]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"{'样本':<12}{'总前景px':>12}{'连通块':>8}{'贴边块':>8}{'贴边px':>12}"
          f"{'碎片<500px':>12}{'碎片px':>12}{'碎片占比':>10}")
    for s in SAMPLES:
        p = MAC / f"{s}-mac.tif"
        if not p.exists():
            print(f"  {s}: 缺文件")
            continue
        a = np.asarray(Image.open(p).convert("L"))
        fg = a < 128
        H, W = fg.shape
        n, lab, st, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), 8)
        edge_px = 0; edge_n = 0; frag_px = 0; frag_n = 0
        tot = int(fg.sum())
        for j in range(1, n):
            x, y, w, h, area = st[j]
            touch = (x <= 0 or y <= 0 or x + w >= W or y + h >= H)
            if touch:
                edge_n += 1; edge_px += int(area)
            elif area < 500:
                frag_n += 1; frag_px += int(area)
        print(f"{s:<12}{tot:>12}{n-1:>8}{edge_n:>8}{edge_px:>12}{frag_n:>12}{frag_px:>12}"
              f"{100*frag_px/max(1,tot):>9.2f}%")
    print(f"\n  -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
