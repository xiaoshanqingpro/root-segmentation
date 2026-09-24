"""生成 WinRHIZO 兼容的 TIFF：补齐 PIL 默认不写的标准标签。

背景（2026-09-23）:
  WinRHIZO 打不开 PIL 写的 TIFF，报 "Cannot read TIFF compressed files [1]"（报错信息误导）。
  实测对比发现: 能打开的（Photoshop 存的）带了 NewSubfileType/Orientation 等标签，
  PIL 写的这些标签全缺。两者压缩方式、条带、尺寸完全一致。

本脚本产出 3 个测试文件供人工在 WinRHIZO 里逐一试:
  A: 补齐标签的未压缩 TIFF
  B: 补齐标签 + 分条带的未压缩 TIFF
  C: JPEG（已知 WinRHIZO 能读，作为兜底）
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, TiffImagePlugin

Image.MAX_IMAGE_PIXELS = None
OUT = Path(r"D:\根系分割项目\outputs\WinRHIZO兼容性测试")
OUT.mkdir(parents=True, exist_ok=True)


def write_tiff(im: Image.Image, dst: Path, rows_per_strip=None, **extra):
    """用显式 IFD 写未压缩 TIFF，补齐 WinRHIZO/Photoshop 会写的标准标签。"""
    ifd = TiffImagePlugin.ImageFileDirectory_v2()
    ifd[254] = 0            # NewSubfileType: 0 = 全分辨率主图
    ifd[274] = 1            # Orientation: 1 = 正常
    ifd[282] = 600.0        # XResolution
    ifd[283] = 600.0        # YResolution
    ifd[296] = 2            # ResolutionUnit: 2 = inch
    ifd[305] = "RootSeg 1.0"   # Software
    ifd[306] = "2026:09:23 00:00:00"   # DateTime
    if rows_per_strip:
        ifd[278] = rows_per_strip
    for k, v in extra.items():
        ifd[int(k)] = v
    im.save(dst, format="TIFF", tiffinfo=ifd, compression=None, dpi=(600, 600))


def main() -> int:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        (Path.home() / "Desktop" / "测试" / "机器" / "281-003-mac.tif")
    stem = src.stem.replace("-mac", "")
    print(f"源: {src}")
    im = Image.open(src).convert("RGB")
    arr = np.asarray(im)

    a = OUT / f"{stem}_A_补齐标签.tif"
    write_tiff(im, a)
    print(f"  A 补齐标签 未压缩单条带 : {a.name}  {a.stat().st_size/1024**2:.1f} MB")

    b = OUT / f"{stem}_B_补齐标签_分条带.tif"
    write_tiff(im, b, rows_per_strip=16)
    print(f"  B 补齐标签 未压缩分条带 : {b.name}  {b.stat().st_size/1024**2:.1f} MB")

    c = OUT / f"{stem}_C_jpeg.jpg"
    Image.fromarray(arr).save(c, "JPEG", quality=98, subsampling=0, dpi=(600, 600))
    print(f"  C JPEG q98（兜底方案）  : {c.name}  {c.stat().st_size/1024**2:.1f} MB")

    # 打印 A 的标签，确认补齐成功
    with Image.open(a) as t:
        tg = t.tag_v2
        print("\n  A 的标签: " + ", ".join(
            f"{k}={tg[k]}" for k in (254, 274, 278, 282, 283, 296, 305, 306) if k in tg))
    print(f"\n  测试文件目录: {OUT}")
    print("  请在 WinRHIZO 里依次试 A → B → C，告诉我哪个能打开。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
