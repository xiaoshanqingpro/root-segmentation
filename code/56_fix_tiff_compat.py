"""批量修复 TIFF 兼容性：把 PIL 写的 TIFF 重存成 WinRHIZO 可读的格式。

问题: PIL 写的 TIFF 缺 NewSubfileType/Orientation 等标签 -> WinRHIZO 拒收。
修法: 用 tiffsave.save_tiff_winrhizo 重存（未压缩 + 补齐标签）。

用法:
    python 56_fix_tiff_compat.py <目录或文件> [--mode winrhizo|lzw|jpeg] [--dry-run]

  --mode winrhizo（默认）: 未压缩 + 补齐标签  -> WinRHIZO 可读，体积大
  --mode jpeg            : JPEG q98            -> WinRHIZO 可读，体积小（约 2MB）
  --mode lzw             : LZW 压缩 + 补齐标签 -> 仅编辑/存档用，WinRHIZO 读不了
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import paths as P  # noqa: E402
import tiffsave as TS  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
EXTS = (".tif", ".tiff")


def already_ok(p: Path) -> bool:
    """是否已经带 NewSubfileType 标签（说明是补齐过的或是 Photoshop 存的）"""
    try:
        with Image.open(p) as im:
            return im.tag_v2.get(254) is not None
    except Exception:  # noqa: BLE001
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("--mode", choices=["winrhizo", "lzw", "jpeg"], default="winrhizo")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="连已带标签的也重存")
    a = ap.parse_args()

    t = Path(a.target)
    files = sorted(t.rglob("*")) if t.is_dir() else [t]
    files = [p for p in files if p.suffix.lower() in EXTS]
    print(f"[修复] 目标 {t}   找到 {len(files)} 个 TIFF   模式 {a.mode}")

    todo = [p for p in files if a.force or not already_ok(p)]
    skip = len(files) - len(todo)
    print(f"[修复] 待处理 {len(todo)} 个（已带标签跳过 {skip} 个）")
    if a.dry_run:
        for p in todo[:20]:
            print(f"  {p}")
        return 0

    t0 = time.time()
    before = after = 0
    fail = []
    for i, p in enumerate(todo, 1):
        try:
            b = p.stat().st_size
            before += b
            im = Image.open(p).convert("RGB")
            if a.mode == "jpeg":
                dst = p.with_suffix(".jpg")
                TS.save_jpeg(im, dst)
                p.unlink()                      # 原 TIFF 删掉，避免同名混淆
            else:
                tmp = p.with_suffix(".tmp.tif")
                TS.save_tiff_winrhizo(im, tmp, compress=(a.mode == "lzw"))
                tmp.replace(p)
                dst = p
            after += dst.stat().st_size
            if i % 20 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"  [{i}/{len(todo)}] {el:.0f}s  {el/i:.1f}s/个  "
                      f"累计 {after/1024**3:.2f} GB", flush=True)
        except Exception as e:  # noqa: BLE001
            fail.append((str(p), f"{type(e).__name__}: {e}"))
            print(f"  !! 失败 {p.name}: {e}")

    ok = len(todo) - len(fail)
    print(f"\n  完成 {ok}/{len(todo)} 个，用时 {(time.time()-t0)/60:.1f} 分钟")
    if before:
        print(f"  体积 {before/1024**3:.2f} GB -> {after/1024**3:.2f} GB")
    for s, m in fail[:10]:
        print(f"  !! {s}: {m}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
