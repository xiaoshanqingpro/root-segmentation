"""WinRHIZO 兼容的 TIFF 写入（共用模块）。

问题（2026-09-23 实测）:
  PIL 默认写出的 TIFF 缺 NewSubfileType / Orientation 等标准标签，
  WinRHIZO 会拒绝打开，并报**误导性**错误 "Cannot read TIFF compressed files [1]"
  ——文件其实根本没压缩（Compression=1）。

  对照证据:
    7-003-man.tif      (Photoshop 存, 标签齐全, 未压缩)  -> WinRHIZO 能打开 ✅
    281-003-mac.tif    (PIL 存, 缺标签,   未压缩)        -> WinRHIZO 打不开 ❌
    281-003_A_补齐标签.tif (PIL 存 + 补齐标签, 未压缩)     -> WinRHIZO 能打开 ✅（用户实测）

用法:
    from tiffsave import save_tiff_winrhizo, save_jpeg
    save_tiff_winrhizo(img, "out.tif")          # 未压缩 + 补齐标签（WinRHIZO 可读）
    save_tiff_winrhizo(img, "out.tif", compress=True)   # LZW 压缩（体积小，WinRHIZO 读不了）
"""
from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
from PIL import Image, TiffImagePlugin

SOFTWARE = "RootSeg 1.0"


def save_tiff_winrhizo(im: Image.Image | np.ndarray, dst: str | Path,
                       dpi: int = 600, compress: bool = False,
                       rows_per_strip: int | None = 16) -> Path:
    """写 TIFF 并补齐 WinRHIZO 要求的标签。

    compress=False（默认）: 未压缩 —— WinRHIZO 可读
    compress=True         : LZW 压缩 —— 体积小得多，但 WinRHIZO 读不了（用于编辑/存档）
    """
    if isinstance(im, np.ndarray):
        im = Image.fromarray(im)
    im = im.convert("RGB")
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    ifd = TiffImagePlugin.ImageFileDirectory_v2()
    ifd[254] = 0                                   # NewSubfileType: 全分辨率主图
    ifd[274] = 1                                   # Orientation: 正常
    ifd[282] = float(dpi)                          # XResolution
    ifd[283] = float(dpi)                          # YResolution
    ifd[296] = 2                                   # ResolutionUnit: inch
    ifd[305] = SOFTWARE                            # Software
    ifd[306] = datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S")
    if rows_per_strip:
        ifd[278] = rows_per_strip

    im.save(dst, format="TIFF", tiffinfo=ifd,
            compression=("tiff_lzw" if compress else None), dpi=(dpi, dpi))
    return dst


def save_jpeg(im: Image.Image | np.ndarray, dst: str | Path,
              dpi: int = 600, quality: int = 98) -> Path:
    """JPEG：WinRHIZO 与 Photoshop 都能读，体积小（约 2 MB）。
    实测 WinRHIZO 可正常读取本管线产出的 JPEG。"""
    if isinstance(im, np.ndarray):
        im = Image.fromarray(im)
    im = im.convert("RGB")
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst, "JPEG", quality=quality, subsampling=0, dpi=(dpi, dpi))
    return dst
