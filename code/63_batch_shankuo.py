"""杉阔混交林 全量生成机器图（自包含，不依赖 importlib 加载数字开头的脚本）。

输出: 根系分割项目\\机器图_杉阔混交林\\<原结构>\\<样本>-mac.tif
规则与 `59_batch_all.py` 完全一致：同一套冻结参数 + 同一个 residue 清理。

注意:
  - `9.8校准\\` 那 35 张与 `扫描\\原始图片\\9.8校准` **MD5 完全相同**，结果必然相同；
    为保证本数据集输出自成一套、结构与源目录对应，这里仍全部重跑。
  - `F\\` 里有 34 张与 9.8校准 同名（同一物理样本的旋转版），**属重复样本**，记在清单里。
  - `F\\` 里 14 张是 300dpi (2481x3509)，会被重采样到 600dpi。
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
SRC = Path.home() / "Desktop" / "杉阔混交林"
OUT = P.PROJ / "机器图_杉阔混交林"
TARGET_DPI = 600.0
WORKERS = 8
EXTS = (".jpg", ".jpeg", ".tif", ".tiff", ".png")


def set_params():
    R.FEAT_LONG = 3000
    R.HALO_GROW = 0
    R.SHRINK_PX = 1
    R.SMOOTH_ALL = True
    R.SMOOTH_OPEN_PX = 1
    R.BRIDGE_CLOSE_PX = 2


def worker(task):
    src_s, dst_s = task
    set_params()
    import cv2
    cv2.setNumThreads(1)
    t0 = time.time()
    src, dst = Path(src_s), Path(dst_s)
    im = Image.open(src)
    dpi = float((im.info.get("dpi") or (600, 600))[0])
    rgb = np.asarray(im.convert("RGB"))
    resampled = False
    if abs(dpi - TARGET_DPI) > 1:
        k = TARGET_DPI / dpi
        rgb = np.asarray(Image.fromarray(rgb).resize(
            (max(1, round(rgb.shape[1] * k)), max(1, round(rgb.shape[0] * k))), Image.LANCZOS))
        resampled = True
    root, _, info = R.extract_root(rgb, return_maps=False)
    img = np.where(root, 0, 255).astype(np.uint8)
    img, se = RES.clean_residue(img)
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).convert("RGB").save(
        dst, format="TIFF", compression="raw", dpi=(600, 600))
    return {"文件": src.name, "输出": str(dst.relative_to(OUT)),
            "源dpi": int(round(dpi)), "重采样": resampled,
            "尺寸": f"{img.shape[1]}x{img.shape[0]}",
            "根占比%": round(float((img == 0).mean() * 100), 3),
            "删贴边": se.get("edge", 0) + se.get("edge_strip", 0) + se.get("edge_iso", 0)
                      + se.get("edge_margin", 0),
            "删碎屑": se.get("debris", 0) + se.get("blob", 0),
            "保留近块": se.get("kept", 0),
            "耗时s": round(time.time() - t0, 1)}


def main() -> int:
    set_params()
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in SRC.rglob("*") if p.suffix.lower() in EXTS)
    tasks = [(str(p), str(OUT / p.relative_to(SRC).parent / (p.stem + "-mac.tif")))
             for p in files]
    print(f"[杉阔混交林] {len(tasks)} 张 -> {OUT}")
    from collections import Counter
    print("  分目录:", dict(Counter(str(p.relative_to(SRC).parent) for p in files)))

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
                print(f"    {i}/{len(tasks)}  已用 {el/60:.1f} 分钟  "
                      f"均 {el/i:.1f}s/张  剩余约 {el/i*(len(tasks)-i)/60:.1f} 分钟", flush=True)

    with (OUT / "_批量清单.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        if rows:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)

    ok = [r for r in rows if not str(r.get("输出", "")).startswith("ERROR")]
    print(f"\n  成功 {len(ok)}/{len(rows)}")
    if ok:
        import statistics as st
        v = [float(r["根占比%"]) for r in ok]
        print(f"  根占比 中位 {st.median(v):.3f}%  最小 {min(v):.3f}%  最大 {max(v):.3f}%")
        print(f"  重采样 {sum(1 for r in ok if r['重采样'])} 张")
        low = sorted([r for r in ok if float(r["根占比%"]) < 0.3], key=lambda r: float(r["根占比%"]))
        if low:
            print(f"  根占比 <0.3% 的 {len(low)} 张（需复核）:")
            for r in low[:12]:
                print(f"    {r['输出']:<44} {float(r['根占比%']):.3f}%")
    for r in rows:
        if str(r.get("输出", "")).startswith("ERROR"):
            print(f"  !! {r}")
    print(f"  清单 -> {OUT/'_批量清单.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
