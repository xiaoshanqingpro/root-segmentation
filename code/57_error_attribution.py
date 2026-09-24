"""误差归因分类：阴影太深 vs 根太淡，各出一条证据链，全部写进 测试\\调试\\。

做法:
  1. 对 12 对样本的原图算形态学特征（根灰度、阴影占比、根粗细、根占比）
  2. 与端到端误差合并
  3. 按"阴影占比"和"根灰度"分成两组，分别做误差统计（bootstrap 95% CI）
  4. 每组挑一条最有代表性的样本，出"原图 / 人工修图 / 机器修图 + 数据"证据图
  5. 全部产物写进 C:\\Users\\<用户名>\\Desktop\\测试\\调试\\
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
import rootseg_common as R  # noqa: E402
import paths as P  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

Image.MAX_IMAGE_PIXELS = None
DBG = P.TEST_DBG                     # 测试\调试\
AN = P.OUTPUTS / "误差分析"
METRICS = [("Length(cm)", "根总长", "cm"), ("SurfArea(cm2)", "表面积", "cm²"),
           ("RootVolume(cm3)", "根体积", "cm³"), ("AvgDiam(mm)", "平均直径", "mm")]

R.FEAT_LONG = 3000
R.HALO_GROW = 0
R.SHRINK_PX = 1


def font(sz, b=False):
    for n in (("msyhbd.ttc", "msyh.ttc") if b else ("msyh.ttc",)):
        try:
            return ImageFont.truetype(n, sz)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


def find_orig(stem: str) -> Path | None:
    for root in (P.RAW_D2, P.RAW_D1):
        if not root.exists():
            continue
        for e in (".jpg", ".jpeg", ".tif", ".tiff", ".png"):
            hits = [p for p in root.rglob(stem + e) if "9.9改" not in p.parts]
            if hits:
                return hits[0]
    return None


def features(stem: str) -> dict:
    src = find_orig(stem)
    if src is None:
        return {}
    rgb = np.asarray(Image.open(src).convert("RGB"))
    H, W = rgb.shape[:2]
    sc = min(1.0, 1600 / max(H, W))
    small = cv2.resize(rgb, (int(W * sc), int(H * sc)), interpolation=cv2.INTER_AREA)
    hh, ww = small.shape[:2]
    fg = ~R.flood_background(small)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1].astype(np.float32) / 255.0
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32)
    k = max(31, (int(min(hh, ww) * 0.05) // 2) * 2 + 1)
    illum = cv2.morphologyEx(gray.astype(np.uint8), cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(np.float32)
    depth = illum - gray
    deep = depth > R.DARK_DROP
    core = fg & deep & ((sat >= R.SAT_ROOT) | (gray < R.GRAY_TUB))
    if core.sum() < 50:
        core = fg & deep
    shadow = fg & deep & ~core
    dt = cv2.distanceTransform(core.astype(np.uint8), cv2.DIST_L2, 5)
    return {"原图": str(src.relative_to(P.SCAN)),
            "根灰度": round(float(np.median(gray[core])), 1),
            "根饱和度": round(float(np.median(sat[core])), 3),
            "根宽度px": round(float(np.median(dt[core]) / max(sc, 1e-6)) * 2, 2),
            "阴影占比%": round(100 * float(shadow.mean()), 3),
            "根占比%": round(100 * float(core.mean()), 3)}


def boot_ci(x, n=10000, seed=42):
    x = np.asarray([v for v in x if np.isfinite(v)], dtype=float)
    if len(x) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    m = np.array([np.median(rng.choice(x, len(x), replace=True)) for _ in range(n)])
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def main() -> int:
    DBG.mkdir(parents=True, exist_ok=True)
    pairs = list(csv.DictReader((AN / "pairs.csv").open(encoding="utf-8-sig")))
    print(f"载入 {len(pairs)} 对样本")

    rows = []
    for r in pairs:
        s = r["样本"]
        f = features(s)
        rec = {"样本": s, **f}
        for k, cn, _ in METRICS:
            rec[f"误差%_{cn}"] = float(r.get(f"误差%_{cn}") or "nan")
        # 四项绝对误差的中位，作为"整体偏差"代表
        vals = [abs(rec[f"误差%_{cn}"]) for _, cn, _ in METRICS
                if np.isfinite(rec[f"误差%_{cn}"])]
        rec["四项绝对误差中位"] = round(float(np.median(vals)), 2) if vals else float("nan")
        rows.append(rec)
        print(f"  {s:<12} 灰度 {rec.get('根灰度','?'):>5} 阴影 {rec.get('阴影占比%','?'):>6}%  "
              f"宽 {rec.get('根宽度px','?'):>5}px  四项绝对误差中位 {rec['四项绝对误差中位']:>5.1f}%")

    keys = list(rows[0].keys())
    with (AN / "误差归因.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)

    # ---- 分组: 按阴影占比高 / 根灰度大(淡) ----
    sh = np.array([r.get("阴影占比%", np.nan) for r in rows], dtype=float)
    gy = np.array([r.get("根灰度", np.nan) for r in rows], dtype=float)
    rows_sorted_sh = sorted(rows, key=lambda r: -r.get("阴影占比%", 0))
    rows_sorted_gy = sorted(rows, key=lambda r: -r.get("根灰度", 0))
    n_grp = max(3, len(rows) // 3)
    grp_sh = rows_sorted_sh[:n_grp]
    grp_gy = rows_sorted_gy[:n_grp]
    print(f"\n阴影太深组 ({n_grp} 张): {[r['样本'] for r in grp_sh]}")
    print(f"根太淡组   ({n_grp} 张): {[r['样本'] for r in grp_gy]}")

    def grp_stat(grp, tag):
        out = {}
        for _, cn, _ in METRICS:
            v = [r[f"误差%_{cn}"] for r in grp if np.isfinite(r[f"误差%_{cn}"])]
            if not v:
                continue
            lo, hi = boot_ci(v)
            out[cn] = {"n": len(v), "median": round(float(np.median(v)), 2),
                       "ci95": [round(lo, 2), round(hi, 2)],
                       "median_abs": round(float(np.median(np.abs(v))), 2)}
        return out

    stat_sh, stat_gy = grp_stat(grp_sh, "阴影"), grp_stat(grp_gy, "淡根")
    print(f"\n{'指标':<10}{'阴影太深组 中位[95%CI]':>34}{'根太淡组 中位[95%CI]':>34}")
    for _, cn, _ in METRICS:
        a, b = stat_sh.get(cn, {}), stat_gy.get(cn, {})
        sa = f"{a.get('median',float('nan')):+.1f}% [{a.get('ci95',[0,0])[0]:+.1f},{a.get('ci95',[0,0])[1]:+.1f}]"
        sb = f"{b.get('median',float('nan')):+.1f}% [{b.get('ci95',[0,0])[0]:+.1f},{b.get('ci95',[0,0])[1]:+.1f}]"
        print(f"{cn:<10}{sa:>34}{sb:>34}")

    (AN / "归因分组.json").write_text(json.dumps(
        {"n_grp": n_grp, "阴影太深组": [r["样本"] for r in grp_sh],
         "根太淡组": [r["样本"] for r in grp_gy],
         "阴影组统计": stat_sh, "淡根组统计": stat_gy},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  -> {AN}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
