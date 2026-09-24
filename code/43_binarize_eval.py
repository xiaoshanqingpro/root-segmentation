"""二值化阈值敏感性分析 + 用人工真值评估当前管线。

用户要求:
  - 二值化，只留黑白，人工图与机器图叠加对比像素点
  - 留意人工图可能有灰色噪点

要点:
  阈值定低了灰噪会被当成根；定高了淡细根被抹掉。
  先扫阈值看"人工掩膜面积"对阈值的敏感度，找平缓区间；
  再在同一阈值下比较人工掩膜与机器掩膜。
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import rootseg_common as R  # noqa: E402

# --- 统一路径（见 paths.py；数据搬家只改那里）---
sys.path.insert(0, str(Path(__file__).parent))
import paths as P  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
PROJ = P.PROJ
MAN = P.TEST_MAN
ORIG = P.TEST_ORIG
OUT = PROJ / "outputs" / "二值化分析"
SAMPLES = ["66-003", "28-CL-003", "G218-001", "126-003", "134-003", "7-003"]
THRS = [100, 120, 128, 140, 150, 160, 180, 200, 220]


def orig_of(stem):
    for e in (".tif", ".jpg", ".png"):
        p = ORIG / (stem + e)
        if p.exists():
            return p
    return None


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    # ---- 1. 阈值敏感性 ----
    print("=== 人工掩膜面积(%) 随二值化阈值的变化 ===")
    hdr = "样本".ljust(12) + "".join(f"{t:>8}" for t in THRS)
    print(hdr)
    sens = {}
    for s in SAMPLES:
        p = MAN / f"{s}-man.tif"
        if not p.exists():
            continue
        g = np.asarray(Image.open(p).convert("L"))
        row = [100.0 * float((g < t).mean()) for t in THRS]
        sens[s] = row
        print(s.ljust(12) + "".join(f"{v:>8.3f}" for v in row))

    # 选阈值: 相邻阈值间面积相对变化最小的那一段的中值
    if sens:
        arr = np.array([sens[s] for s in sens])
        rel = np.abs(np.diff(arr, axis=1)) / np.maximum(arr[:, :-1], 1e-6)
        mean_rel = rel.mean(0)
        print("\n各阈值区间的平均相对变化率:")
        for i, t in enumerate(THRS[:-1]):
            print(f"  {t:>4} -> {THRS[i+1]:<4}: {mean_rel[i]*100:5.2f}%")
        best_i = int(np.argmin(mean_rel))
        THR = (THRS[best_i] + THRS[best_i + 1]) // 2
        print(f"\n  最平缓的区间: {THRS[best_i]}~{THRS[best_i+1]}  ->  取阈值 {THR}")
    else:
        THR = 150

    # ---- 2. 用该阈值比较人工 vs 机器 ----
    print(f"\n=== 人工 vs 机器（统一二值化阈值 {THR}）===")
    rows = []
    print(f"{'样本':<12}{'人工%':>9}{'机器%':>9}{'倍数':>8}{'IoU':>8}{'召回':>8}{'精确':>8}")
    for s in SAMPLES:
        pm = MAN / f"{s}-man.tif"
        po = orig_of(s)
        if not pm.exists() or po is None:
            print(f"  {s}: 缺文件")
            continue
        man = np.asarray(Image.open(pm).convert("L")) < THR
        rgb = np.asarray(Image.open(po).convert("RGB"))
        root, _, info = R.extract_root(rgb, return_maps=False)
        if root.shape != man.shape:
            print(f"  {s}: 尺寸不一致 {root.shape} vs {man.shape}")
            continue
        inter = int((man & root).sum())
        iou = inter / max(1, int((man | root).sum()))
        rec = inter / max(1, int(man.sum()))
        pre = inter / max(1, int(root.sum()))
        rows.append({"样本": s, "阈值": THR,
                     "人工根面积%": round(100 * man.mean(), 4),
                     "机器根面积%": round(100 * root.mean(), 4),
                     "面积倍数": round(root.mean() / max(1e-9, man.mean()), 3),
                     "IoU": round(iou, 4), "召回": round(rec, 4), "精确": round(pre, 4),
                     "漏检%": round(100 * (1 - rec), 2), "误检%": round(100 * (1 - pre), 2)})
        print(f"{s:<12}{100*man.mean():>9.3f}{100*root.mean():>9.3f}"
              f"{root.mean()/max(1e-9,man.mean()):>8.3f}{iou:>8.4f}{rec:>8.4f}{pre:>8.4f}")

    if rows:
        with (OUT / "人工vs机器.csv").open("w", newline="", encoding="utf-8-sig") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
        import statistics as st
        print("\n  中位: " + "  ".join(
            f"{k} {st.median(r[k] for r in rows):.4f}" for k in ["IoU", "召回", "精确"]))
        print(f"  中位面积倍数 {st.median(r['面积倍数'] for r in rows):.3f}")
    with (OUT / "阈值敏感性.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.writer(fh)
        wr.writerow(["样本"] + [f"thr{t}" for t in THRS])
        for s, v in sens.items():
            wr.writerow([s] + v)
    print(f"\n  -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
