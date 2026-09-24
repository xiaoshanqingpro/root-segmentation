"""训练数据专项审计：找出所有 人工(-man) / 机器(-mac) / 原图 的配对关系。

用户说明（2026-09-23）:
  - 文件名带 `-man` 后缀 = **人工标注**
  - 文件名带 `-mac` 后缀 = **机器产出**
  - 训练用**两类（根 / 非根）**，且已有端到端（WinRHIZO）数据

本脚本只读扫描，产出配对清单供训练使用。
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
DESKTOP = Path(r"C:\Users\LIHAOYANG\Desktop")
OUT = Path(r"D:\根系分割项目\outputs\训练数据审计")

SEARCH_ROOTS = [
    DESKTOP / "测试",
    DESKTOP / "扫描" / "根系分割项目",
    DESKTOP / "扫描" / "原始图片",
]
EXTS = (".tif", ".tiff", ".jpg", ".jpeg", ".png")


def stem_of(p: Path) -> str:
    """去掉 -man / -mac / -9.22 等后缀，得到样本号"""
    s = p.stem
    s = re.sub(r"-(man|mac)(-\d+(\.\d+)?)?$", "", s)
    s = re.sub(r"-(man|mac)$", "", s)
    return s


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    man, mac, other = {}, {}, []
    seen = set()
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.suffix.lower() not in EXTS or p in seen:
                continue
            seen.add(p)
            name = p.stem.lower()
            if "-man" in name:
                man.setdefault(stem_of(p), []).append(p)
            elif "-mac" in name:
                mac.setdefault(stem_of(p), []).append(p)
            else:
                other.append(p)

    print("=" * 70)
    print(f"1. 人工标注 (-man)   共 {sum(len(v) for v in man.values())} 个文件 / "
          f"{len(man)} 个样本")
    print("=" * 70)
    for k in sorted(man):
        for p in man[k]:
            with Image.open(p) as im:
                sz = im.size
            print(f"  {k:<14} {p.name:<26} {sz[0]}x{sz[1]}  "
                  f"{p.stat().st_size/1024**2:7.1f} MB  {p.parent}")

    print()
    print("=" * 70)
    print(f"2. 机器产出 (-mac)   共 {sum(len(v) for v in mac.values())} 个文件 / "
          f"{len(mac)} 个样本")
    print("=" * 70)
    for k in sorted(mac):
        for p in mac[k]:
            with Image.open(p) as im:
                sz = im.size
            print(f"  {k:<14} {p.name:<26} {sz[0]}x{sz[1]}  "
                  f"{p.stat().st_size/1024**2:7.1f} MB  {p.parent}")

    print()
    print("=" * 70)
    print("3. 配对情况")
    print("=" * 70)
    both = sorted(set(man) & set(mac))
    only_man = sorted(set(man) - set(mac))
    only_mac = sorted(set(mac) - set(man))
    print(f"  同时有人工+机器: {len(both)} 个样本  {both}")
    print(f"  只有人工:        {len(only_man)} 个  {only_man}")
    print(f"  只有机器:        {len(only_mac)} 个  {only_mac}")

    # 4. 找每个样本的原图
    print()
    print("=" * 70)
    print("4. 原图匹配（按样本号在 测试\\原图 与 原始图片 里找）")
    print("=" * 70)
    orig_dirs = [DESKTOP / "测试" / "原图", DESKTOP / "扫描" / "图片",
                 DESKTOP / "扫描" / "原始图片"]
    orig_index = {}
    for d in orig_dirs:
        if not d.exists():
            continue
        for p in d.rglob("*"):
            if p.suffix.lower() in EXTS and "-man" not in p.stem.lower() \
                    and "-mac" not in p.stem.lower():
                orig_index.setdefault(p.stem, []).append(p)

    rows = []
    for k in sorted(set(man) | set(mac)):
        o = orig_index.get(k, [])
        rows.append({
            "样本": k,
            "人工": ";".join(str(p) for p in man.get(k, [])),
            "机器": ";".join(str(p) for p in mac.get(k, [])),
            "原图数": len(o),
            "原图": ";".join(str(p) for p in o[:2]),
        })
        flag = "" if o else "   <-- 找不到原图！"
        print(f"  {k:<14} 人工{len(man.get(k,[]))} 机器{len(mac.get(k,[]))} "
              f"原图{len(o)}{flag}")

    with (OUT / "training_pairs.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    # 5. 人工图上取二值标签的可行性
    print()
    print("=" * 70)
    print("5. 人工图二值化检查（非白=根）")
    print("=" * 70)
    for k in sorted(man):
        p = man[k][0]
        a = np.asarray(Image.open(p).convert("L"))
        nw = a < 250
        print(f"  {k:<14} {p.name:<26} 根像素 {100*nw.mean():6.3f}%   "
              f"灰度 p50(根)={np.median(a[nw]) if nw.any() else -1:.0f}")

    print(f"\n  清单 -> {OUT/'training_pairs.csv'}")
    return 0


if __name__ == "__main__":
    main()
