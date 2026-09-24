"""第二数据集处理：判根 → 非根转纯白。**多进程并行**。

用户决策（2026-09-22）:
  1. 9.15 重复的两份都保留，**读取时按 MD5 去重**
  2. 9.9 用**原图**；9.9改（手工修过的）**整体排除、不碰**
  3. 朝向**不转**，保持横向 7019x4962
  4. **不裁剪图片**，用原图大小
  5. 非根全部转纯白；**按原来的组织编号放进新建文件夹**

判根逻辑与数据集1 共用 `rootseg_common.py`，保证口径一致。
第二数据集特殊性: 四边有扫描仪黑边，洪水填充补"内部亮点种子"（已内置）。

用法:
    python 25_process_dataset2.py --dry-run
    python 25_process_dataset2.py --names 22-001 G120-001
    python 25_process_dataset2.py                    # 全量
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import rootseg_common as R  # noqa: E402

# --- 统一路径（见 paths.py；数据搬家只改那里）---
sys.path.insert(0, str(Path(__file__).parent))
import paths as P  # noqa: E402

PROJ = P.PROJ
SRC = P.RAW_D2
OUT_DIR = PROJ / "白底根图_第二数据集"
PREVIEW_DIR = PROJ / "outputs" / "白底根图_第二数据集_preview"
REPORT = PROJ / "outputs" / "白底根图_第二数据集_report.csv"
MANIFEST = PROJ / "outputs" / "白底根图_第二数据集_manifest.json"

EXCLUDE_DIRS = {"9.9改", "work", "ocr_raw", "__pycache__"}
EXTS = (".jpg", ".jpeg", ".tif", ".tiff", ".png")
DEDUPE = True
WORKERS = 8


def md5(p: Path, buf: int = 1 << 22) -> str:
    h = hashlib.md5()
    with p.open("rb") as fh:
        while True:
            b = fh.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def collect_files():
    files = sorted(p for p in SRC.rglob("*") if p.suffix.lower() in EXTS)
    kept, skipped, seen = [], [], {}
    for p in files:
        rel = p.relative_to(SRC)
        if any(part in EXCLUDE_DIRS for part in rel.parts):
            skipped.append({"rel": str(rel), "why": "目录在排除清单（9.9改/工作目录）"})
            continue
        if DEDUPE:
            h = md5(p)
            if h in seen:
                skipped.append({"rel": str(rel),
                                "why": f"内容重复，与 {seen[h].relative_to(SRC)} 相同", "md5": h})
                continue
            seen[h] = p
        kept.append(p)
    return kept, skipped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--names", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=WORKERS)
    args = ap.parse_args()

    kept, skipped = collect_files()
    if args.names:
        kept = [p for p in kept if p.stem in set(args.names)]

    print(f"[数据集2] 源 {SRC}")
    print(f"[数据集2] 去重后 {len(kept)} 张待处理；跳过 {len(skipped)} 张")
    for s in skipped[:6]:
        print(f"    跳过: {s['rel']}   ({s['why']})")
    if len(skipped) > 6:
        print(f"    … 另有 {len(skipped)-6} 张")
    if args.dry_run:
        from collections import Counter
        print("  按目录:", dict(Counter(str(p.relative_to(SRC).parent) for p in kept)))
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    tasks = []
    for p in kept:
        rel = p.relative_to(SRC)
        tasks.append((str(p), str(OUT_DIR / rel.parent / (p.stem + ".jpg")),
                      str(PREVIEW_DIR / f"{rel.parent.name}__{p.stem}.png"), True))

    print(f"[数据集2] 并行 {args.workers} 进程")
    t0 = time.time()
    rows = R.run_parallel(tasks, workers=args.workers, label="数据集2")
    el = time.time() - t0

    ok = [r for r in rows if "error" not in r]
    ok.sort(key=lambda r: r["file"])
    if ok:
        with REPORT.open("w", newline="", encoding="utf-8-sig") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(ok[0].keys()))
            wr.writeheader()
            wr.writerows(ok)
    MANIFEST.write_text(json.dumps({
        "src": str(SRC), "out": str(OUT_DIR), "crop": False, "rotate": False,
        "excluded_dirs": sorted(EXCLUDE_DIRS), "deduped": skipped, "n_out": len(ok),
        "rule": {"flood_tol": R.FLOOD_TOL, "feat_long": R.FEAT_LONG, "gray_tub": R.GRAY_TUB,
                 "min_area_feat": R.MIN_AREA_FEAT, "pale_rescue": R.PALE_RESCUE},
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n  完成 {len(ok)}/{len(rows)} 张，用时 {el/60:.1f} 分钟（{el/max(1,len(rows)):.1f}s/张）")
    if ok:
        import statistics as st
        print(f"  根占比 中位 {st.median(r['root_pct'] for r in ok):.3f}%   "
              f"灰底残留 中位 {st.median(r['graybg_pct'] for r in ok):.1f}%   "
              f"碎块 中位 {st.median(r['tiny_components'] for r in ok):.0f} 个")
    for r in [r for r in rows if "error" in r][:5]:
        print(f"  !! 失败 {r['file']}: {r['error']}")
    print(f"  白底根图 -> {OUT_DIR}\n  报告 -> {REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
