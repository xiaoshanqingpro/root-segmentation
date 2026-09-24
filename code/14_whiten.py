"""白底根图生成（数据集1）：判根 → 非根全部转纯白。**多进程并行**。

判根逻辑统一在 `rootseg_common.py`，数据集1 / 数据集2 共用同一套参数。

2026-09-22 修订:
  用户反馈"有灰底残留、有小碎片、非根没全变白" -> 见 rootseg_common.py 顶部说明
  用户反馈"GPU/CPU 占用都不高，能不能快点" -> 改为多进程并行（本机 32 逻辑核，
  但可用内存只有 ~4.6GB，所以并行度默认 8，避免爆内存）

输入: processed/600dpi/   （已裁边 + 统一 600dpi）
产出: 白底根图/<批次>/<原文件名>
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import rootseg_common as R  # noqa: E402

# --- 统一路径（见 paths.py；数据搬家只改那里）---
sys.path.insert(0, str(Path(__file__).parent))
import paths as P  # noqa: E402

PROJ = P.PROJ
SRC_DIR = PROJ / "processed" / "600dpi"
OUT_DIR = PROJ / "白底根图"
PREVIEW_DIR = PROJ / "outputs" / "白底根图_preview"
REPORT = PROJ / "outputs" / "白底根图_report.csv"
WORKERS = 8


def main() -> int:
    only = [a for a in sys.argv[1:] if not a.startswith("-")]
    files = sorted(SRC_DIR.rglob("*.jpg"))
    if only:
        files = [p for p in files if p.stem in only]
    print(f"[白底根图·数据集1] {len(files)} 张  并行 {WORKERS} 进程")
    print(f"[白底根图·数据集1] 容差 {R.FLOOD_TOL}  特征尺度 {R.FEAT_LONG}  "
          f"管状需够暗 <{R.GRAY_TUB}  最小块 {R.MIN_AREA_FEAT}px")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    tasks = [(str(p), str(OUT_DIR / p.parent.name / p.name),
              str(PREVIEW_DIR / f"{p.stem}_preview.png"), True) for p in files]

    t0 = time.time()
    rows = R.run_parallel(tasks, workers=WORKERS, label="数据集1")
    el = time.time() - t0

    rows.sort(key=lambda r: r.get("file", ""))
    ok = [r for r in rows if "error" not in r]
    with REPORT.open("w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(ok[0].keys()))
        wr.writeheader()
        wr.writerows(ok)

    print(f"\n  完成 {len(ok)}/{len(rows)} 张，用时 {el/60:.1f} 分钟（{el/max(1,len(rows)):.1f}s/张）")
    if ok:
        import statistics as st
        print(f"  根占比 中位 {st.median(r['root_pct'] for r in ok):.3f}%   "
              f"灰底残留 中位 {st.median(r['graybg_pct'] for r in ok):.1f}%   "
              f"碎块 中位 {st.median(r['tiny_components'] for r in ok):.0f} 个")
    bad = [r for r in rows if "error" in r]
    for r in bad[:5]:
        print(f"  !! 失败 {r['file']}: {r['error']}")
    print(f"  白底根图 -> {OUT_DIR}\n  报告 -> {REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
