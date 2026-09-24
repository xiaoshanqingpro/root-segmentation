"""参数优化：用 6 张人工真值扫"收边 / 抗锯齿 / 接断点"，找最优后处理组合。

效率做法: 每张图只算一次特征，拿到"后处理之前"的原始掩膜，
         再对同一份原始掩膜反复施加不同的后处理组合 —— 省掉重复的特征计算。

评价口径（按用户看重的指标）:
  IoU / 精确 / 召回        掩膜重合度
  面积倍数                 -> 直接决定 表面积、根体积
  宽度偏差                 -> 直接决定 直径
  骨架长度比               -> 直接决定 根总长
"""
from __future__ import annotations

import csv
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
PROJ = P.PROJ
MAN = P.TEST_MAN
ORIG = P.TEST_ORIG
OUT = PROJ / "outputs" / "参数优化"
SAMPLES = ["66-003", "28-CL-003", "G218-001", "126-003", "134-003", "7-003"]
THR = 210

HALOS = [0, 1]
SHRINKS = [0, 1, 2, 3]
OPENS = [0, 1]
CLOSES = [0, 2]


def orig_of(stem):
    for e in (".tif", ".jpg", ".png"):
        p = ORIG / (stem + e)
        if p.exists():
            return p
    return None


def skeleton_len(mask: np.ndarray) -> float:
    try:
        from skimage.morphology import skeletonize
        sk = skeletonize(mask)
        return float(sk.sum())
    except Exception:  # noqa: BLE001
        return float("nan")


def width_median(mask: np.ndarray) -> float:
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    v = dt[mask]
    return float(np.median(v) * 2) if v.size else 0.0


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    # ---- 1. 预先算好: 原始掩膜(未后处理) + 人工掩膜 + 人工的宽度/骨架 ----
    cache = []
    for s in SAMPLES:
        pm, po = MAN / f"{s}-man.tif", orig_of(s)
        if not pm.exists() or po is None:
            print(f"  跳过 {s}: 缺文件")
            continue
        man = np.asarray(Image.open(pm).convert("L")) < THR
        rgb = np.asarray(Image.open(po).convert("RGB"))
        R.HALO_GROW = 0
        raw, _, _ = R.extract_root(rgb, return_maps=False, apply_post=False)
        if raw.shape != man.shape:
            print(f"  跳过 {s}: 尺寸不一致")
            continue
        cache.append({"s": s, "man": man, "raw": raw,
                      "man_w": width_median(man), "man_len": skeleton_len(man),
                      "man_area": float(man.mean())})
        print(f"  载入 {s}: 人工 {100*man.mean():.3f}%  宽 {cache[-1]['man_w']:.2f}px  "
              f"骨架 {cache[-1]['man_len']:.0f}px", flush=True)

    if not cache:
        print("无可用样本")
        return 1

    # ---- 2. 扫后处理组合 ----
    print(f"\n{'HALO':>5}{'SHRINK':>7}{'OPEN':>6}{'CLOSE':>7}{'IoU':>8}{'召回':>8}{'精确':>8}"
          f"{'面积倍数':>9}{'宽度比':>8}{'长度比':>8}")
    rows = []
    for halo in HALOS:
        for shr in SHRINKS:
            for op in OPENS:
                for cl in CLOSES:
                    ks = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * shr + 1,) * 2) if shr else None
                    ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * op + 1,) * 2) if op else None
                    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * cl + 1,) * 2) if cl else None
                    ious, recs, pres, ar, wr, lr = [], [], [], [], [], []
                    for c in cache:
                        m = c["raw"]
                        if ks is not None:
                            m = cv2.erode(m.astype(np.uint8), ks) > 0
                        if kc is not None:
                            m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, kc) > 0
                        if ko is not None:
                            m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, ko) > 0
                        man = c["man"]
                        inter = int((man & m).sum())
                        ious.append(inter / max(1, int((man | m).sum())))
                        recs.append(inter / max(1, int(man.sum())))
                        pres.append(inter / max(1, int(m.sum())))
                        ar.append(m.mean() / max(1e-9, c["man_area"]))
                        w = width_median(m)
                        wr.append(w / max(1e-9, c["man_w"]))
                        lr.append(skeleton_len(m) / max(1e-9, c["man_len"]))
                    r = {"halo": halo, "shrink": shr, "open": op, "close": cl,
                         "IoU": round(float(np.mean(ious)), 4),
                         "recall": round(float(np.mean(recs)), 4),
                         "precision": round(float(np.mean(pres)), 4),
                         "面积倍数": round(float(np.mean(ar)), 3),
                         "宽度比": round(float(np.mean(wr)), 3),
                         "长度比": round(float(np.mean(lr)), 3)}
                    rows.append(r)
                    print(f"{halo:>5}{shr:>7}{op:>6}{cl:>7}{r['IoU']:>8.4f}{r['recall']:>8.4f}"
                          f"{r['precision']:>8.4f}{r['面积倍数']:>9.3f}{r['宽度比']:>8.3f}"
                          f"{r['长度比']:>8.3f}", flush=True)

    with (OUT / "参数扫描.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    # ---- 3. 综合评分: IoU 高 + 面积/宽度/长度都接近 1 ----
    def score(r):
        return (r["IoU"]
                - 0.5 * abs(r["面积倍数"] - 1)
                - 0.3 * abs(r["宽度比"] - 1)
                - 0.3 * abs(r["长度比"] - 1))
    best = max(rows, key=score)
    print(f"\n  综合最优: HALO={best['halo']} SHRINK={best['shrink']} "
          f"OPEN={best['open']} CLOSE={best['close']}")
    print(f"    IoU {best['IoU']:.4f}  召回 {best['recall']:.4f}  精确 {best['precision']:.4f}")
    print(f"    面积倍数 {best['面积倍数']:.3f}  宽度比 {best['宽度比']:.3f}  长度比 {best['长度比']:.3f}")
    print(f"  表 -> {OUT/'参数扫描.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
