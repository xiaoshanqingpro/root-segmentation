"""阶段一 · 预处理：按图裁掉边缘伪影带 + 统一重采样到 600 dpi。

依据（用户已确认的四项决策）:
  1. 类别设计 = 三类（根/阴影/背景）
  2. 装包 → 脚本 00 验证
  3. 左边缘那条粉色刻度尺是**无意带进来的**，裁掉；且**处理后图片水平/垂直分辨率必须是 600 dpi**
  4. 裁剪策略 = **按图单独裁**，用实测带宽（edge_band.csv），不做统一裁剪

本脚本做三件事:
  a) 逐张按 edge_band.csv 的四边实测带宽裁掉伪影带（向上取整到 0.5mm，再加安全余量）
  b) 把非 600 dpi 的图重采样到 600 dpi（0814 批次是 720 dpi，缩放系数 600/720）
  c) 写出时写入 600 dpi 元数据，JPEG 4:4:4 无色度下采样（避免破坏"粉色尺/低饱和阴影"的判据）

原图只读、绝不修改。被排除的图（透射背光件、树叶）单独放，不进主数据集。

用法:
    python 09_preprocess.py --dry-run     # 只报尺寸与体积，不写文件
    python 09_preprocess.py               # 真正写出
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path.home() / "Desktop" / "扫描"
PROJ = ROOT / "根系分割项目"
IMAGE_DIR = ROOT / "图片"
STAGE1 = PROJ / "outputs" / "阶段一"
OUT_MAIN = PROJ / "processed" / "600dpi"
OUT_EXCL = PROJ / "processed" / "600dpi_excluded"

TARGET_DPI = 600.0
SAFETY_MM = 0.5          # 实测带宽之外的额外安全余量
MAX_CROP_MM = 15.0       # 单边裁剪上限（防误检把整张图裁没）
JPEG_QUALITY = 95

# 用户已确认移出主数据集
EXCLUDED = {
    "图片\\树根004.jpg": "透射背光成像（黑底橙根）+ 带刻度尺，与其余白底接触式扫描不同源",
    "图片\\树叶003.jpg": "内容为叶片，非根系",
}


def load_band_table() -> dict[str, dict]:
    path = STAGE1 / "edge_band.csv"
    if not path.exists():
        raise SystemExit(f"缺少 {path}，请先运行 08_edge_band.py")
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    return {r["rel_path"]: r for r in rows}


def round_up_half_mm(mm: float) -> float:
    return float(np.ceil(mm * 2) / 2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只统计，不写文件")
    args = ap.parse_args()

    bands = load_band_table()
    files = sorted(IMAGE_DIR.rglob("*.jpg"))
    print(f"[preprocess] 源图 {len(files)} 张 | 目标 {TARGET_DPI:.0f} dpi | "
          f"安全余量 {SAFETY_MM}mm | {'DRY-RUN' if args.dry_run else '写出'}")

    manifest, total_bytes, n_main, n_excl, n_scaled = [], 0, 0, 0, 0
    for p in files:
        rel = str(p.relative_to(ROOT))
        with Image.open(p) as raw:
            w0, h0 = raw.size
            dpi = float((raw.info.get("dpi") or (600, 600))[0])

        mmpp0 = 25.4 / dpi
        rec = {"rel_path": rel, "src_w": w0, "src_h": h0, "src_dpi": dpi,
               "src_mm_w": round(w0 * mmpp0, 2), "src_mm_h": round(h0 * mmpp0, 2),
               "excluded": rel in EXCLUDED, "exclude_reason": EXCLUDED.get(rel, "")}

        # ---- a) 逐张裁剪 ----
        band = bands.get(rel)
        crops = {"left": 0.0, "right": 0.0, "top": 0.0, "bottom": 0.0}
        if band and not rec["excluded"]:
            for side in crops:
                mm = round_up_half_mm(float(band[f"{side}_mm"])) + (SAFETY_MM if float(band[f"{side}_mm"]) > 0 else 0.0)
                crops[side] = min(mm, MAX_CROP_MM)
        px = {s: int(round(mm / mmpp0)) for s, mm in crops.items()}

        # 保底：裁完至少留 50% 画幅
        if (px["left"] + px["right"]) > w0 * 0.5 or (px["top"] + px["bottom"]) > h0 * 0.5:
            for s in px:
                px[s] = 0
            crops = {s: 0.0 for s in crops}
            rec["crop_aborted"] = True

        w1, h1 = w0 - px["left"] - px["right"], h0 - px["top"] - px["bottom"]

        # ---- b) 重采样到 600 dpi ----
        mmpp_target = 25.4 / TARGET_DPI
        w2 = int(round(w1 * mmpp0 / mmpp_target))
        h2 = int(round(h1 * mmpp0 / mmpp_target))
        if (w2, h2) != (w1, h1):
            n_scaled += 1

        rec.update({
            "crop_left_mm": crops["left"], "crop_right_mm": crops["right"],
            "crop_top_mm": crops["top"], "crop_bottom_mm": crops["bottom"],
            "out_w": w2, "out_h": h2, "out_dpi": int(TARGET_DPI),
            "out_mm_w": round(w2 * mmpp_target, 2), "out_mm_h": round(h2 * mmpp_target, 2),
            "resampled": (w2, h2) != (w1, h1),
            "resample_factor": round(mmpp0 / mmpp_target, 6),
        })

        out_dir = OUT_EXCL if rec["excluded"] else OUT_MAIN
        batch = p.parent.name if p.parent != IMAGE_DIR else "_根目录"
        out_path = out_dir / batch / p.name
        rec["out_path"] = str(out_path.relative_to(PROJ))

        if not args.dry_run:
            im = Image.open(p)
            im.draft("RGB", (w1, h1))          # 让解码直接降到裁剪后尺度，省内存
            im = im.convert("RGB")
            if im.size != (w0, h0):
                im = im.resize((w0, h0), Image.LANCZOS)   # draft 只近似，尺寸要对上裁剪坐标
            if any(px.values()):
                im = im.crop((px["left"], px["top"], w0 - px["right"], h0 - px["bottom"]))
            if (w2, h2) != im.size:
                im = im.resize((w2, h2), Image.LANCZOS)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            im.save(out_path, "JPEG", quality=JPEG_QUALITY, subsampling=0, dpi=(TARGET_DPI, TARGET_DPI))
            rec["out_bytes"] = out_path.stat().st_size
        else:
            rec["out_bytes"] = int(w2 * h2 * 0.55)      # 粗略估计
        total_bytes += rec["out_bytes"]

        manifest.append(rec)
        n_excl += int(rec["excluded"])
        n_main += int(not rec["excluded"])

    OUT = STAGE1 / "preprocess_manifest.csv"
    if not args.dry_run:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        keys = list(manifest[0].keys())
        with OUT.open("w", newline="", encoding="utf-8-sig") as fh:
            wr = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            wr.writeheader()
            wr.writerows(manifest)

    sizes = {(r["out_w"], r["out_h"]) for r in manifest if not r["excluded"]}
    cropped = [r for r in manifest if not r["excluded"] and any(
        r[f"crop_{s}_mm"] > 0 for s in ["left", "right", "top", "bottom"])]
    print(f"\n  主数据集 {n_main} 张 | 排除 {n_excl} 张 | 发生重采样 {n_scaled} 张")
    print(f"  裁过边的图: {len(cropped)} 张")
    print(f"  输出尺寸种类: {sorted(sizes)}")
    print(f"  预计/实际占用: {total_bytes/1024**3:.2f} GB  ->  {OUT_MAIN if not args.dry_run else '(dry-run)'}")
    if not args.dry_run:
        print(f"  清单: {OUT.relative_to(PROJ)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
