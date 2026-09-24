"""检查 杉阔混交林 数据集：尺寸/DPI/是否与已有数据集重复。"""
from __future__ import annotations

import collections
import hashlib
import sys
from pathlib import Path

from PIL import Image

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

Image.MAX_IMAGE_PIXELS = None
D = Path.home() / "Desktop" / "杉阔混交林"
OLD998 = Path.home() / "Desktop" / "扫描" / "原始图片" / "9.8校准"
MACHINE = Path(r"D:\根系分割项目\机器图")


def md5(p: Path, buf: int = 1 << 22) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(buf), b""):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    for sub in ["9.8校准", "F", "F补"]:
        fs = sorted((D / sub).glob("*.jpg"))
        sizes, dpis = collections.Counter(), collections.Counter()
        for p in fs:
            with Image.open(p) as im:
                sizes[im.size] += 1
                d = im.info.get("dpi")
                dpis[round(d[0]) if d else None] += 1
        print(f"{sub:<8} {len(fs):>3} 张   尺寸 {dict(sizes)}   dpi {dict(dpis)}")
        print(f"         例: {[p.name for p in fs[:5]]}")

    print("\n=== 与已有数据集查重 ===")
    a = {md5(p): p.name for p in (D / "9.8校准").glob("*.jpg")}
    if OLD998.exists():
        b = {md5(p): p.name for p in OLD998.glob("*.jpg")}
        print(f"  杉阔混交林\\9.8校准 唯一 {len(a)}   扫描\\原始图片\\9.8校准 唯一 {len(b)}")
        print(f"  两边 MD5 相同: {len(set(a) & set(b))}  <- 相同则说明是同一批图")

    # 与机器图目录里已有的比（文件名层面）
    have = {p.stem for p in MACHINE.rglob("*-mac.tif")}
    allnew = [p.stem for p in D.rglob("*.jpg")]
    dup = [s for s in allnew if s in have]
    print(f"\n  与已生成机器图同名的: {len(dup)} / {len(allnew)}")
    if dup:
        print(f"    例: {dup[:10]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
