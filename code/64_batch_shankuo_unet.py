"""杉阔混交林 · U-Net 版机器图（与在跑的经典管线批次互不干扰）。

背景:
  另有一个 `63_batch_shankuo.py` 正在跑（经典规则管线 rootseg_common，8 进程 CPU），
  输出到 `机器图_杉阔混交林\\<结构>\\<样本>-mac.tif`。
  本脚本改用 **U-Net**，因此**必须写到不同目录**，否则同名文件会互相覆盖。
  输出: `机器图_杉阔混交林_unet\\<原结构>\\<样本>-mac.tif`

处理要点:
  1. **DPI 归一**：`F\\` 里有 14 张是 300 dpi(2481x3509)，U-Net 在 600 dpi 上训练，
     不归一的话细根像素宽度减半，会系统性漏检 -> 统一重采样到 600 dpi
  2. **MD5 去重**：跳过与已处理内容重复的文件（记入清单，不算失败）
  3. U-Net 滑窗推理 -> 边框自动清理 -> 细壁闭合环剔除 -> 非根涂白
  4. 输出 600 dpi TIFF，根保留原色

用法:
    python 64_batch_shankuo_unet.py --dry-run
    python 64_batch_shankuo_unet.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as P  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
SRC = Path(r"C:\Users\LIHAOYANG\Desktop\杉阔混交林")
OUT = P.PROJ / "机器图_杉阔混交林_unet"
EXTS = (".jpg", ".jpeg", ".tif", ".tiff", ".png")
TARGET_DPI = 600.0
NONWHITE = 250


def md5(p: Path, buf: int = 1 << 22) -> str:
    h = hashlib.md5()
    with p.open("rb") as fh:
        while True:
            b = fh.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_dpi_ok(p: Path) -> tuple[np.ndarray, float]:
    """读图并归一到 600 dpi。返回 (rgb, 原始dpi)"""
    im = Image.open(p)
    d = im.info.get("dpi")
    src_dpi = float(d[0]) if d else TARGET_DPI
    a = np.asarray(im.convert("RGB"))
    if abs(src_dpi - TARGET_DPI) > 1:
        k = TARGET_DPI / src_dpi
        import cv2
        a = cv2.resize(a, (int(round(a.shape[1] * k)), int(round(a.shape[0] * k))),
                       interpolation=cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA)
    return a, src_dpi


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    files = sorted(p for p in SRC.rglob("*") if p.suffix.lower() in EXTS)
    print(f"[U-Net 杉阔混交林] 源 {SRC}   {len(files)} 张")
    if a.limit:
        files = files[:a.limit]

    # 预扫：DPI 分布 + MD5 去重
    specs, seen = [], {}
    for p in files:
        im = Image.open(p)
        d = im.info.get("dpi")
        dpi = round(float(d[0])) if d else -1
        size = im.size
        im.close()
        h = md5(p)
        dup = h in seen
        specs.append({"p": p, "dpi": dpi, "size": size, "md5": h, "dup_of": seen.get(h)})
        seen.setdefault(h, p)
    from collections import Counter
    size_ct = Counter("{0}x{1}".format(s["size"][0], s["size"][1]) for s in specs)
    print(f"  DPI 分布: {dict(Counter(s['dpi'] for s in specs))}")
    print(f"  尺寸分布: {dict(size_ct)}")
    ndup = sum(1 for s in specs if s["dup_of"])
    print(f"  内容重复: {ndup} 张（将跳过，记入清单）")
    if a.dry_run:
        for s in specs[:10]:
            print(f"    {s['p'].relative_to(SRC)}  dpi={s['dpi']}  {s['size']}"
                  + (f"  [重复于 {s['dup_of'].name}]" if s["dup_of"] else ""))
        return 0

    # 载入 U-Net
    import torch
    import segmentation_models_pytorch as smp
    import borderclean as BC
    import importlib.util as _il
    spec = _il.spec_from_file_location("pa", str(Path(__file__).parent / "49_preannotate.py"))
    W = P.PROJ / "outputs" / "训练" / "best_unet.pt"
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(W, map_location="cpu", weights_only=False)
    model = smp.Unet(encoder_name=ck["encoder"], encoder_weights=None,
                     in_channels=3, classes=1).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"  U-Net {ck['encoder']} 载入完成（训练时 val dice {ck['val']['dice']:.4f}）  device={dev}")

    # 复用 49 的推理与细线清理
    ns = {"__name__": "pa", "__file__": str(Path(__file__).parent / "49_preannotate.py")}
    src49 = (Path(__file__).parent / "49_preannotate.py").read_text(encoding="utf-8")
    exec(compile(src49.split("def main(")[0], "pa", "exec"), ns)
    unet_mask, remove_thin_loops = ns["unet_mask"], ns["remove_thin_loops"]

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    todo = [s for s in specs if not s["dup_of"]]
    print(f"  实际处理 {len(todo)} 张（跳过重复 {ndup} 张）")
    t0 = time.time()
    for i, s in enumerate(todo, 1):
        p = s["p"]
        rel = p.relative_to(SRC)
        img, src_dpi = load_dpi_ok(p)
        mask = unet_mask(model, dev, img)
        out = img.copy()
        out[~mask] = 255
        out, bc = BC.clean_borders(out)
        out, n_loops = remove_thin_loops(out)
        dst = OUT / rel.parent / (p.stem + "-mac.tif")
        dst.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(out).save(dst, dpi=(600, 600))
        fg = float(np.any(out < NONWHITE, axis=2).mean() * 100)
        rows.append({"rel": str(rel), "out": str(dst.relative_to(P.PROJ)),
                     "src_dpi": src_dpi, "src_w": s["size"][0], "src_h": s["size"][1],
                     "out_w": out.shape[1], "out_h": out.shape[0],
                     "root_pct": round(fg, 3),
                     "removed_border_px": bc["removed_total_px"],
                     "removed_loops": n_loops})
        if i % 5 == 0 or i == len(todo):
            el = time.time() - t0
            print(f"  [{i}/{len(todo)}] {rel}  dpi{s['dpi']}  根 {fg:6.3f}%  "
                  f"边框{bc['removed_total_px']:>7}px  环{n_loops:>3}  "
                  f"已用 {el:.0f}s  ETA {el/i*(len(todo)-i):.0f}s", flush=True)

    with (OUT / "batch_unet_report.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        keys = list(rows[0].keys()) if rows else ["rel"]
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    if ndup:
        with (OUT / "duplicates.csv").open("w", newline="", encoding="utf-8-sig") as fh:
            wr = csv.writer(fh)
            wr.writerow(["dup_file", "same_as"])
            for s in specs:
                if s["dup_of"]:
                    wr.writerow([str(s["p"].relative_to(SRC)), str(s["dup_of"].relative_to(SRC))])
    print(f"\n  完成 {len(rows)} 张，用时 {(time.time()-t0)/60:.1f} 分钟")
    print(f"  输出 -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
