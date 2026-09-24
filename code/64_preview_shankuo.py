"""杉阔混交林 复核预览：原图 vs 机器图，缩小后并排输出，供目视判断。

用法:
    python 64_preview_shankuo.py <样本名...>
生成: D:\\根系分割项目\\outputs\\杉阔复核\\<样本>_对比.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import paths as P  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

Image.MAX_IMAGE_PIXELS = None
SRC = Path.home() / "Desktop" / "杉阔混交林"
MACH = P.PROJ / "机器图_杉阔混交林"
OUT = P.PROJ / "outputs" / "杉阔复核"
EXTS = (".jpg", ".jpeg", ".tif", ".tiff", ".png")
TARGET_W = 900


def find(stem: str) -> tuple[Path | None, Path | None]:
    src = next((p for p in SRC.rglob("*") if p.stem == stem and p.suffix.lower() in EXTS), None)
    mac = next((p for p in MACH.rglob("*") if p.stem == stem + "-mac" and p.suffix.lower() == ".tif"), None)
    return src, mac


def thumb(p: Path, w: int = TARGET_W) -> Image.Image:
    im = Image.open(p).convert("RGB")
    h = max(1, round(im.height * w / im.width))
    return im.resize((w, h), Image.LANCZOS)


def main(stems: list[str]) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for s in stems:
        src, mac = find(s)
        if src is None:
            print(f"  !! 找不到原图 {s}")
            continue
        a = thumb(src)
        if mac is None:
            print(f"  !! 找不到机器图 {s}")
            b = Image.new("RGB", a.size, (200, 200, 200))
        else:
            b = thumb(mac)
        # 原图 + 机器图 并排，中间 8px 分隔
        canvas = Image.new("RGB", (a.width + b.width + 8, max(a.height, b.height)), (255, 0, 0))
        canvas.paste(a, (0, 0))
        canvas.paste(b, (a.width + 8, 0))
        dst = OUT / f"{s}_对比.png"
        canvas.save(dst)
        # 根占比
        if mac is not None:
            m = np.asarray(Image.open(mac).convert("L"))
            ratio = float((m < 128).mean() * 100)
        else:
            ratio = float("nan")
        print(f"  {s:<20} 原图 {a.width}x{a.height}  机器图根占比 {ratio:.3f}%  -> {dst.name}")
    print(f"\n输出目录: {OUT}")
    return 0


if __name__ == "__main__":
    args = sys.argv[1:] or [
        "12-MZ-003", "23-B-GG-002", "16-CL-001", "32-MZ-001", "16-NS-003",
    ]
    sys.exit(main(args))
