"""区分「游离碎片」与「断掉的细根」——决定该删还是该留。

关键问题: 小连通块到底是纸屑灰尘（该删），还是被管线打断的细根（该留/该接）？
判据: 看它离主体根系有多远。
  贴主体（近距离） -> 大概率是断掉的细根 -> 应保留甚至接回
  远离主体         -> 纸屑/灰尘/误判斑点 -> 应删除
再统计贴边黑块的形态（是否沿边成带）。

产出: outputs/残留分析/碎片分类.csv + 建议阈值
"""
from __future__ import annotations

import csv
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

BIG = 2000          # 面积 >= 此值算"主体"
DIST_BINS = [10, 20, 40, 80, 160, 10 ** 9]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"{'样本':<12}{'主体块':>7}{'主体px':>11}{'小块':>7}"
          f"{'<=10px':>10}{'10-20':>9}{'20-40':>9}{'40-80':>9}{'80-160':>9}{'>160':>9}")
    for s in SAMPLES:
        p = MAC / f"{s}-mac.tif"
        if not p.exists():
            continue
        a = np.asarray(Image.open(p).convert("L"))
        fg = a < 128
        H, W = fg.shape
        n, lab, st, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), 8)
        big = [j for j in range(1, n) if st[j, cv2.CC_STAT_AREA] >= BIG]
        small = [j for j in range(1, n) if st[j, cv2.CC_STAT_AREA] < BIG]
        main = np.isin(lab, big)
        main_px = int(main.sum())
        # 到主体的距离场
        dist = cv2.distanceTransform((~main).astype(np.uint8), cv2.DIST_L2, 5)
        bins = [0] * (len(DIST_BINS))
        dists = []
        for j in small:
            m = lab == j
            d = float(dist[m].min())          # 该块到主体的最近距离
            dists.append((int(st[j, cv2.CC_STAT_AREA]), d))
            for bi, ub in enumerate(DIST_BINS):
                if d <= ub:
                    bins[bi] += 1
                    break
        rows.append({"样本": s, "主体块数": len(big), "主体px": main_px, "小块数": len(small),
                     **{f"距离<={DIST_BINS[i]}px": bins[i] for i in range(len(DIST_BINS))},
                     "小块总面积": int(sum(d[0] for d in dists))})
        print(f"{s:<12}{len(big):>7}{main_px:>11}{len(small):>7}"
              + "".join(f"{b:>9}" for b in bins))

    keys = list(rows[0].keys())
    with (OUT / "碎片分类.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)

    print("\n  读数: 若小块绝大多数落在 '<=10px' 或 '10-20px' 档，")
    print("        说明它们是**贴着主体的碎屑/断根**，不能用'孤立小碎块'一刀切删。")
    print(f"\n  -> {OUT/'碎片分类.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
