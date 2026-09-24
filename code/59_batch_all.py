"""全量生成机器图（验收通过后的正式批量）。

产物: 根系分割项目\\机器图\\
  ├─ 数据集1\\<原结构>\\<样本>-mac.tif
  └─ 数据集2\\<原结构>\\<样本>-mac.tif

规则（与验收批次完全一致）:
  - 判根: rootseg_common 冻结参数（FEAT_LONG=3000, HALO=0, SHRINK=1, 开1/闭2）
  - 后处理: residue.clean_residue（含"硬余量"规则，能清掉不接触边界的扫描边）
  - 输出: 纯黑白（根=0, 背景=255）、未压缩 TIFF、600 dpi
  - 不裁剪、不转朝向，保持原图尺寸（非 600dpi 的先重采样到 600dpi）
  - 数据集2 按 MD5 去重（9.15 有两份完全相同的拷贝）
  - 数据集1 排除两个非样本件: 树叶003.jpg、树根004.jpg（透射背光扫描）
"""
from __future__ import annotations

import csv
import hashlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import rootseg_common as R  # noqa: E402
import residue as RES  # noqa: E402
import paths as P  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

Image.MAX_IMAGE_PIXELS = None

OUT_ROOT = P.PROJ / "机器图"
TARGET_DPI = 600.0
WORKERS = 8
EXTS = (".jpg", ".jpeg", ".tif", ".tiff", ".png")

# 数据集1 里不属于本研究的两个文件
EXCLUDE_D1 = {"树叶003.jpg", "树根004.jpg"}

# 冻结参数（在每个子进程里也要设一遍）
def _set_params():
    R.FEAT_LONG = 3000
    R.HALO_GROW = 0
    R.SHRINK_PX = 1
    R.SMOOTH_ALL = True
    R.SMOOTH_OPEN_PX = 1
    R.BRIDGE_CLOSE_PX = 2


def md5(p: Path, buf: int = 1 << 22) -> str:
    h = hashlib.md5()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def collect(src: Path, out_root: Path, exclude_names: set[str], dedupe: bool):
    """收集待处理文件 -> (src, dst) 任务，保持相对结构。"""
    files = sorted(p for p in src.rglob("*") if p.suffix.lower() in EXTS)
    tasks, skipped, seen = [], [], {}
    for p in files:
        rel = p.relative_to(src)
        if p.name in exclude_names:
            skipped.append({"文件": str(rel), "原因": "排除清单"})
            continue
        if dedupe:
            h = md5(p)
            if h in seen:
                skipped.append({"文件": str(rel), "原因": f"内容重复，同 {seen[h].relative_to(src)}"})
                continue
            seen[h] = p
        dst = out_root / rel.parent / (p.stem + "-mac.tif")
        tasks.append((str(p), str(dst)))
    return tasks, skipped


def worker(task):
    src_s, dst_s = task
    _set_params()
    import cv2
    cv2.setNumThreads(1)
    t0 = time.time()
    src, dst = Path(src_s), Path(dst_s)
    im = Image.open(src)
    dpi = float((im.info.get("dpi") or (600, 600))[0])
    rgb = np.asarray(im.convert("RGB"))
    resampled = False
    if abs(dpi - TARGET_DPI) > 1:                 # 非 600dpi: 先重采样到 600dpi
        k = TARGET_DPI / dpi
        # 用 round 而不是 int: 5954 * 600/720 = 4961.67，int 会给 4961，
        # 与同批 600dpi 的 4962 差 1px，造成尺寸不一致。
        rgb = np.asarray(Image.fromarray(rgb).resize(
            (max(1, round(rgb.shape[1] * k)), max(1, round(rgb.shape[0] * k))), Image.LANCZOS))
        resampled = True
    root, _, info = R.extract_root(rgb, return_maps=False)
    img = np.where(root, 0, 255).astype(np.uint8)
    img, se = RES.clean_residue(img)
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).convert("RGB").save(
        dst, format="TIFF", compression="raw", dpi=(600, 600))
    return {"文件": src.name, "输出": str(dst.relative_to(OUT_ROOT)),
            "源dpi": int(round(dpi)), "重采样": resampled,
            "尺寸": f"{img.shape[1]}x{img.shape[0]}",
            "根占比%": round(float((img == 0).mean() * 100), 3),
            "删贴边": se.get("edge", 0) + se.get("edge_strip", 0) + se.get("edge_iso", 0)
                      + se.get("edge_margin", 0),
            "删碎屑": se.get("debris", 0) + se.get("blob", 0),
            "保留近块": se.get("kept", 0),
            "耗时s": round(time.time() - t0, 1)}


def run(jobs, label):
    tasks = []
    all_skipped = []
    for src, out, excl, dedupe, name in jobs:
        t, sk = collect(src, out, excl, dedupe)
        print(f"  [{name}] 待处理 {len(t)} 张，跳过 {len(sk)} 张")
        for s in sk:
            all_skipped.append({"数据集": name, **s})
        tasks += t
    if not tasks:
        return [], all_skipped
    print(f"  合计 {len(tasks)} 张，并行 {WORKERS} 进程")
    rows, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(worker, t) for t in tasks]
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                rows.append(fu.result())
            except Exception as exc:  # noqa: BLE001
                rows.append({"文件": "?", "输出": f"ERROR {type(exc).__name__}: {exc}"})
            if i % 20 == 0 or i == len(tasks):
                el = time.time() - t0
                print(f"    {label} {i}/{len(tasks)}  已用 {el/60:.1f} 分钟  "
                      f"均 {el/i:.1f}s/张  剩余约 {el/i*(len(tasks)-i)/60:.1f} 分钟", flush=True)
    return rows, all_skipped


def main() -> int:
    _set_params()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"[全量] 输出根目录 {OUT_ROOT}")
    jobs = [
        (P.RAW_D1, OUT_ROOT / "数据集1", EXCLUDE_D1, False, "数据集1"),
        (P.RAW_D2, OUT_ROOT / "数据集2", set(), True, "数据集2"),
    ]
    rows, skipped = run(jobs, "全量")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUT_ROOT / "_批量清单.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        if rows:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
    with (OUT_ROOT / "_跳过清单.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        if skipped:
            wr = csv.DictWriter(fh, fieldnames=list(skipped[0].keys()))
            wr.writeheader()
            wr.writerows(skipped)

    ok = [r for r in rows if not str(r.get("输出", "")).startswith("ERROR")]
    print(f"\n  成功 {len(ok)}/{len(rows)} 张；跳过 {len(skipped)} 张")
    if ok:
        import statistics as st
        print(f"  根占比 中位 {st.median(r['根占比%'] for r in ok):.3f}%")
        print(f"  总耗时 {sum(r['耗时s'] for r in ok)/60:.1f} 分钟（单进程累计）")
    for r in rows:
        if str(r.get("输出", "")).startswith("ERROR"):
            print(f"  !! {r}")
    print(f"  清单 -> {OUT_ROOT/'_批量清单.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
