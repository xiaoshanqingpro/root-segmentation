"""仔细查 杉阔混交林 与已有数据的重叠，确定真正要跑的是哪些。"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

from PIL import Image

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

Image.MAX_IMAGE_PIXELS = None
NEW = Path(r"C:\Users\LIHAOYANG\Desktop\杉阔混交林")
RAW2 = Path(r"C:\Users\LIHAOYANG\Desktop\扫描\原始图片")
RAW1 = Path(r"C:\Users\LIHAOYANG\Desktop\扫描\图片")
MACHINE = Path(r"D:\根系分割项目\机器图")
EXTS = (".jpg", ".jpeg", ".tif", ".tiff", ".png")


def md5(p: Path, buf: int = 1 << 22) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(buf), b""):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    new_files = [p for p in NEW.rglob("*") if p.suffix.lower() in EXTS]
    old_files = [p for p in RAW2.rglob("*") if p.suffix.lower() in EXTS]
    old_files += [p for p in RAW1.rglob("*") if p.suffix.lower() in EXTS]
    print(f"杉阔混交林 {len(new_files)} 张；已有数据集 {len(old_files)} 张，正在算 MD5 …")

    new_h = {md5(p): p for p in new_files}
    old_h = {md5(p): p for p in old_files}
    print(f"  去重后: 新 {len(new_h)}  旧 {len(old_h)}")
    dup = set(new_h) & set(old_h)
    print(f"  **MD5 重叠 {len(dup)} 张**")
    # 按子目录统计
    from collections import Counter
    c = Counter()
    for h in dup:
        c[str(new_h[h].relative_to(NEW).parent)] += 1
    for k, v in sorted(c.items()):
        tot = len(list((NEW / k).glob("*"))) if (NEW / k).exists() else 0
        print(f"    {k:<12} {v}/{tot} 与已有数据重复")

    fresh = [new_h[h] for h in new_h if h not in old_h]
    print(f"\n  **需要新跑的 {len(fresh)} 张**")
    cc = Counter(str(p.relative_to(NEW).parent) for p in fresh)
    for k, v in sorted(cc.items()):
        print(f"    {k:<12} {v} 张")

    # 机器图目录里是否已有对应输出（按文件名）
    have = {p.name.replace("-mac.tif", "") for p in MACHINE.rglob("*-mac.tif")}
    miss = [p for p in fresh if p.stem not in have]
    print(f"\n  机器图目录里尚未有的: {len(miss)} 张")
    return 0


if __name__ == "__main__":
    sys.exit(main())
