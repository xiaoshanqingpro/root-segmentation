"""打包"扫描对比包"：挑 3 个样本，把 人工 / CV / U-Net 三版统一成同格式，供 WinRHIZO 扫描对比。

关键: **三个版本必须是同一种格式**，否则格式差异会污染对比结果。
      统一用"未压缩 + 补齐标签"的 TIFF（WinRHIZO 实测可读，见 55_winrhizo_compat.py）。

选样依据（覆盖难度谱，尤其是 CV 已知出问题的场景）:
  G218-001  根极淡，端到端实测 CV 根总长 -36.5%（最差）
  7-003     根系密 + 阴影深
  281-003   CV 在根系密集处留下斑点残留

产出: outputs/扫描对比包/
        <样本>_人工.tif / _CV.tif / _UNet.tif
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import paths as P  # noqa: E402
import tiffsave as TS  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
OUT = P.PROJ / "outputs" / "扫描对比包"
SAMPLES = ["G218-001", "7-003", "281-003"]


def index(dirp: Path, suffixes) -> dict[str, Path]:
    m = {}
    if not dirp.exists():
        return m
    for p in dirp.rglob("*"):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".tif", ".tiff", ".png"):
            continue
        s = p.stem
        for suf in suffixes:
            if suf == "":
                m.setdefault(s, p)
                break
            if s.endswith(suf):
                m.setdefault(s[: -len(suf)], p)
                break
    return m


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cand = P.PROJ / "outputs" / "预标注候选"
    unet = {k: v for k, v in index(cand, ["-unet", "-mac"]).items() if v.parent == cand}
    man = {}
    for d in [P.TEST_MAN, P.TEST_MAN_99GAI, P.PROJ / "人工修复mac图"]:
        man.update(index(d, ["-unet-man", "-man"]))
    cv = {}
    # 注意: 用 setdefault 而非 update —— G218-001 在数据集1(纵向)与数据集2(横向)里各有一张，
    # 若用 update，后面的数据集1会覆盖数据集2，导致 CV 版与人工/U-Net 版尺寸不一致、无法对比。
    for d in [P.WHITE_D2, P.WHITE_D1]:
        for k, v in index(d, [""]).items():
            cv.setdefault(k, v)

    print(f"{'样本':<12}{'人工':<10}{'CV':<10}{'U-Net':<10}")
    ok_samples = []
    for s in SAMPLES:
        has = (s in man, s in cv, s in unet)
        print(f"{s:<12}{'有' if has[0] else '缺':<10}{'有' if has[1] else '缺':<10}"
              f"{'有' if has[2] else '缺':<10}")
        if all(has):
            ok_samples.append(s)
    print(f"\n可打包 {len(ok_samples)} 个: {ok_samples}\n")

    for s in ok_samples:
        # 尺寸一致性检查：三版必须同尺寸，否则扫描对比没有意义
        sizes = {}
        for tag, src in (("人工", man[s]), ("CV", cv[s]), ("UNet", unet[s])):
            with Image.open(src) as im:
                sizes[tag] = im.size
        if len(set(sizes.values())) != 1:
            print(f"  !! {s} 三版尺寸不一致，跳过: {sizes}")
            continue
        for tag, src in (("人工", man[s]), ("CV", cv[s]), ("UNet", unet[s])):
            dst = OUT / f"{s}_{tag}.tif"
            im = Image.open(src).convert("RGB")
            TS.save_tiff_winrhizo(im, dst)      # 统一格式：未压缩 + 补齐标签
            print(f"  {s}_{tag:<5} <- {src.name:<26} -> {dst.name}  "
                  f"{dst.stat().st_size/1024**2:.1f} MB  {sizes[tag]}")

    print(f"\n  对比包 -> {OUT}")
    print("  三个版本已统一为「未压缩 TIFF + 补齐标签」，WinRHIZO 应都能读取。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
