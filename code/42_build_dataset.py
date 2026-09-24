"""构建两类分割训练集：原图 -> 二值标签（根=1 / 非根=0）。

数据来源（用户 2026-09-23 说明）:
  *-man   = 人工标注（根保留原色、背景纯白）-> 标签 = 非白像素
  *-mac   = 机器产出（只用于对比，不进训练）
  原图     = 训练输入

要点:
  1. 逐样本核验 原图 与 人工图 尺寸是否一致（必须像素对齐）
  2. 标签 = 人工图非白像素；同时统计"根灰度"，用于识别未清理干净的样本
  3. 按**整图**划分 train/val，同一张图的 patch 不跨集合（任务书硬要求）
  4. 抽 512x512 patch：有内容的图多抽、纯背景少抽

产出:
  outputs/训练/patches_train.npz , patches_val.npz
  outputs/训练/dataset_report.csv , dataset_report.json
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import paths as P  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
OUT = P.PROJ / "outputs" / "训练"
OUT.mkdir(parents=True, exist_ok=True)

PATCH = 512
SEED = 42
NONWHITE = 250          # 人工图里 < 此值算根
VAL_RATIO = 0.25
MAX_PATCH_PER_IMG = 60  # 每张图最多抽多少 patch
MIN_ROOT_FRAC = 0.002   # patch 内根占比低于此值算"纯背景"，只少量保留

# 候选原图目录（按优先级）
ORIG_DIRS = [P.TEST_ORIG, P.RAW_D2_99, P.RAW_D2, P.RAW_D1]

# 排除清单（数据质量问题，需用户重做标注后再纳入）
EXCLUDE = {
    "00000-001": "人工标注里含大量未清理的浅灰（深色根只占 37.1%，灰度中位 225 ≈ 纸色），"
                 "直接当标签会把'灰色残留'教给模型",
    "7003-scan": "是 WinRHIZO 截图，不是样本",
}


def find_orig(stem: str) -> Path | None:
    for d in ORIG_DIRS:
        if not d.exists():
            continue
        for p in d.rglob("*"):
            if p.suffix.lower() in (".tif", ".tiff", ".jpg", ".jpeg") and p.stem == stem:
                return p
    return None


# 自动清理过的边框版本（由 47_auto_clean_borders.py 产出），存在时优先使用
CLEAN_DIR = P.PROJ / "outputs" / "边框清理" / "cleaned"


def prefer_cleaned(p: Path) -> tuple[Path, bool]:
    """若该样本有清理版，返回清理版并标记 True"""
    stem = p.stem.replace("-man", "").replace("-9.22", "")
    c = CLEAN_DIR / f"{stem}-man-clean.tif"
    return (c, True) if c.exists() else (p, False)


def main() -> int:
    rng = np.random.default_rng(SEED)

    # ---- 收集配对 ----
    pairs = []
    for p in P.TEST_MAN_99GAI.glob("*"):
        if "-man" not in p.stem.lower():
            continue
        stem = p.stem.replace("-man", "").replace("-9.22", "")
        o = find_orig(stem)
        pairs.append({"stem": stem, "man": p, "orig": o})
    # 人工目录下还有别的样本（测试\人工 根目录）
    for p in P.TEST_MAN.glob("*-man*"):
        stem = p.stem.replace("-man", "").replace("-9.22", "")
        if any(q["stem"] == stem for q in pairs):
            continue
        o = find_orig(stem)
        pairs.append({"stem": stem, "man": p, "orig": o})
    pairs.sort(key=lambda d: d["stem"])

    print(f"[数据] 人工标注 {len(pairs)} 个样本")
    rows = []
    ok_pairs = []
    for d in pairs:
        if d["stem"] in EXCLUDE:
            print(f"  -- {d['stem']}: 已排除（{EXCLUDE[d['stem']][:40]}…）")
            continue
        if d["orig"] is None:
            print(f"  !! {d['stem']}: 找不到原图，跳过")
            continue
        with Image.open(d["man"]) as im:
            ms = im.size
        with Image.open(d["orig"]) as im:
            os_ = im.size
        if ms != os_:
            print(f"  !! {d['stem']}: 尺寸不一致 人工{ms} vs 原图{os_}，跳过")
            continue
        a = np.asarray(Image.open(d["man"]).convert("RGB"))
        g = np.asarray(Image.open(d["man"]).convert("L"))
        man_used, was_cleaned = prefer_cleaned(d["man"])
        if was_cleaned:
            a = np.asarray(Image.open(man_used).convert("RGB"))
            g = np.asarray(Image.open(man_used).convert("L"))
            print(f"     （使用自动清理版边框: {man_used.name}）")
        mask = np.any(a < NONWHITE, axis=2)
        dark = mask & (g < 170)
        rows.append({
            "stem": d["stem"], "orig": str(d["orig"]), "man": str(man_used),
            "边框已自动清理": was_cleaned,
            "w": ms[0], "h": ms[1],
            "root_pct": round(100 * mask.mean(), 3),
            "dark_root_pct": round(100 * dark.mean(), 3),
            "light_root_pct": round(100 * (mask & ~dark).mean(), 3),
            "dark_ratio": round(float(dark.sum()) / max(1, int(mask.sum())), 3),
            "gray_p50_root": int(np.median(g[mask])) if mask.any() else -1,
        })
        ok_pairs.append({**d, "size": ms})
        print(f"  {d['stem']:<12} {ms[0]}x{ms[1]}  根 {rows[-1]['root_pct']:6.3f}%  "
              f"其中深色 {rows[-1]['dark_ratio']*100:5.1f}%  灰度中位 {rows[-1]['gray_p50_root']}")

    with (OUT / "dataset_report.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    # ---- 划分 train/val（按整图） ----
    idx = rng.permutation(len(ok_pairs))
    n_val = max(1, int(round(len(ok_pairs) * VAL_RATIO)))
    val_ids = {ok_pairs[i]["stem"] for i in idx[:n_val]}
    print(f"\n[划分] train {len(ok_pairs)-n_val} 张 / val {n_val} 张")
    print(f"       val = {sorted(val_ids)}")

    # ---- 抽 patch ----
    def extract(sample_list, tag):
        X, Y, meta = [], [], []
        for d in sample_list:
            orig = np.asarray(Image.open(d["orig"]).convert("RGB"))
            man_p, _ = prefer_cleaned(d["man"])
            man = np.asarray(Image.open(man_p).convert("RGB"))
            H, W = man.shape[:2]
            mask = np.any(man < NONWHITE, axis=2).astype(np.uint8)
            # 网格候选起点
            ys = list(range(0, max(1, H - PATCH), PATCH * 2))
            xs = list(range(0, max(1, W - PATCH), PATCH * 2))
            cand = [(y, x) for y in ys for x in xs]
            cand = [(y, x) for (y, x) in cand if y + PATCH <= H and x + PATCH <= W]
            if not cand:
                continue
            # 算每个 patch 的根占比
            scored = []
            for (y, x) in cand:
                f = float(mask[y:y + PATCH, x:x + PATCH].mean())
                scored.append((f, y, x))
            scored.sort(key=lambda z: -z[0])
            # 有内容的优先，另留少量纯背景
            content = [s for s in scored if s[0] >= MIN_ROOT_FRAC][:MAX_PATCH_PER_IMG]
            bg = [s for s in scored if s[0] < MIN_ROOT_FRAC][:max(2, MAX_PATCH_PER_IMG // 10)]
            picked = content + bg
            for f, y, x in picked:
                X.append(orig[y:y + PATCH, x:x + PATCH])
                Y.append(mask[y:y + PATCH, x:x + PATCH])
                meta.append(f"{d['stem']}:{y}:{x}")
            print(f"    {tag} {d['stem']:<12} 抽 {len(picked):3d} patch "
                  f"(有内容 {len(content)}, 纯背景 {len(bg)})")
        X = np.stack(X).astype(np.uint8) if X else np.zeros((0, PATCH, PATCH, 3), np.uint8)
        Y = np.stack(Y).astype(np.uint8) if Y else np.zeros((0, PATCH, PATCH), np.uint8)
        np.savez_compressed(OUT / f"patches_{tag}.npz", X=X, Y=Y,
                            meta=np.array(meta, dtype=object))
        print(f"  -> patches_{tag}.npz  X={X.shape} Y={Y.shape} "
              f"({(OUT/f'patches_{tag}.npz').stat().st_size/1024**2:.0f} MB)")
        return X, Y

    tr = [d for d in ok_pairs if d["stem"] not in val_ids]
    va = [d for d in ok_pairs if d["stem"] in val_ids]
    print("\n[抽 patch]")
    Xtr, Ytr = extract(tr, "train")
    Xva, Yva = extract(va, "val")

    summary = {"n_samples": len(ok_pairs), "n_train_img": len(tr), "n_val_img": len(va),
               "val_ids": sorted(val_ids), "patch": PATCH, "seed": SEED,
               "train_patches": int(Xtr.shape[0]), "val_patches": int(Xva.shape[0]),
               "train_root_frac": round(float(Ytr.mean()), 4) if Ytr.size else 0,
               "val_root_frac": round(float(Yva.mean()), 4) if Yva.size else 0,
               "samples": rows}
    (OUT / "dataset_report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  训练 patch {Xtr.shape[0]} 张（根占比 {summary['train_root_frac']*100:.3f}%）")
    print(f"  验证 patch {Xva.shape[0]} 张（根占比 {summary['val_root_frac']*100:.3f}%）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
