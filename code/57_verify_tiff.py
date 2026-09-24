"""核验 TIFF 修复结果：两个目录的标签补齐情况。"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

TARGETS = [
    Path(r"D:\根系分割项目\机器图"),
    (Path.home() / "Desktop" / "测试" / "机器"),
    Path(r"D:\根系分割项目\outputs\预标注候选"),
    Path(r"D:\根系分割项目\人工修复mac图"),
]

for d in TARGETS:
    print(f"\n=== {d} ===")
    if not d.exists():
        print("  不存在")
        continue
    fs = sorted(p for p in d.rglob("*") if p.suffix.lower() in (".tif", ".tiff"))
    if not fs:
        print("  无 TIFF")
        continue
    ok = 0
    comps = {}
    for p in fs:
        try:
            with Image.open(p) as im:
                t = im.tag_v2
                if t.get(254) is not None:
                    ok += 1
                c = t.get(259)
                comps[c] = comps.get(c, 0) + 1
        except Exception:  # noqa: BLE001
            pass
    names = {1: "未压缩", 5: "LZW", 8: "Deflate", 32773: "PackBits"}
    cs = ", ".join(f"{names.get(k, k)}×{v}" for k, v in sorted(comps.items()))
    print(f"  {len(fs)} 个 TIFF  已补齐标签 {ok} 个  压缩: {cs}")
    p = fs[0]
    with Image.open(p) as im:
        t = im.tag_v2
        print(f"  样例 {p.name}: 254={t.get(254)} 274={t.get(274)} "
              f"278={t.get(278)} 305={t.get(305)}  {p.stat().st_size/1024**2:.1f} MB")
