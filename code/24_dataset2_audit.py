"""第二数据集（采样数据\\图片）组织问题体检 —— 只读，不修改对方任何文件。

体检项:
  1. 跨目录内容重复（MD5）—— 训练/验证集泄漏的头号风险
  2. 目录内文件名重复
  3. 9.9 与 9.9改 的重叠（按文件名推断，注意 9.9改 是修过的，MD5 必然不同）
  4. 朝向 / dpi / 尺寸一致性
  5. 命名规范异常
"""
from __future__ import annotations

import csv
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

# --- 统一路径（见 paths.py；数据搬家只改那里）---
sys.path.insert(0, str(Path(__file__).parent))
import paths as P  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
SRC = P.RAW_D2
OUT = Path(r"D:\根系分割项目\outputs\第二数据集体检.csv")


def md5(p: Path, buf: int = 1 << 22) -> str:
    h = hashlib.md5()
    with p.open("rb") as fh:
        while True:
            b = fh.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> int:
    files = sorted(p for p in SRC.rglob("*")
                   if p.suffix.lower() in (".jpg", ".jpeg", ".tif", ".tiff", ".png"))
    print(f"[体检] {len(files)} 张，正在计算 MD5 …", flush=True)
    rows = []
    for i, p in enumerate(files, 1):
        with Image.open(p) as im:
            w, h = im.size
            d = im.info.get("dpi")
        rel = p.relative_to(SRC)
        rows.append({
            "rel_path": str(rel), "group": rel.parts[0] if len(rel.parts) > 1 else "_根目录",
            "sub": rel.parts[1] if len(rel.parts) > 2 else "",
            "name": p.name, "stem": p.stem, "ext": p.suffix.lower(),
            "MB": round(p.stat().st_size / 1024 ** 2, 2),
            "w": w, "h": h, "orient": "横向" if w > h else "纵向",
            "dpi": f"{round(d[0])}" if d else "无",
            "md5": md5(p),
        })
        if i % 50 == 0:
            print(f"  ... {i}/{len(files)}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    print("\n=== 1. 跨目录内容重复（MD5 相同）===")
    by_hash = defaultdict(list)
    for r in rows:
        by_hash[r["md5"]].append(r["rel_path"])
    dups = {k: v for k, v in by_hash.items() if len(v) > 1}
    dup_px = 0
    for k, v in sorted(dups.items(), key=lambda kv: -len(kv[1])):
        dup_px += len(v) - 1
        print(f"  [{len(v)} 份] " + "  ==  ".join(v))
    print(f"  重复组 {len(dups)} 个，冗余副本 {dup_px} 张")

    print("\n=== 2. 目录内文件名重复 ===")
    by_dir_name = defaultdict(list)
    for r in rows:
        by_dir_name[(r["group"], r["sub"], r["name"])].append(r["rel_path"])
    same = {k: v for k, v in by_dir_name.items() if len(v) > 1}
    print(f"  同目录同名: {len(same)} 组" + ("" if same else " （无）"))

    print("\n=== 3. 9.9 与 9.9改 的对应关系 ===")
    g99 = {r["stem"]: r for r in rows if r["group"] == "多样性林" and r["sub"] == "9.9"}
    g99g = {r["stem"]: r for r in rows if r["group"] == "多样性林" and r["sub"] == "9.9改"}
    both = sorted(set(g99) & set(g99g))
    print(f"  9.9: {len(g99)} 张, 9.9改: {len(g99g)} 张, 同名重合: {len(both)} 张")
    for s in both:
        a, b = g99[s], g99g[s]
        same_content = a["md5"] == b["md5"]
        print(f"    {s:<12} 9.9 {a['MB']:>7.1f}MB {a['ext']:<5} | "
              f"9.9改 {b['MB']:>7.1f}MB {b['ext']:<5} | 内容{'相同' if same_content else '不同(修过)'}")
    only99g = sorted(set(g99g) - set(g99))
    if only99g:
        print(f"  只在 9.9改 里、9.9 里没有的: {only99g}")

    print("\n=== 4. 朝向 / dpi ===")
    print(f"  朝向: {dict(Counter(r['orient'] for r in rows))}")
    print(f"  dpi : {dict(Counter(r['dpi'] for r in rows))}")
    print(f"  尺寸: {dict(Counter(f'{r[chr(119)]}x{r[chr(104)]}' for r in rows))}")

    print("\n=== 5. 命名规范 ===")
    import re
    pat = re.compile(r"^[0-9A-Za-z]+(-[A-Za-z0-9]+)*(-\d{3})?$")
    odd = [r for r in rows if not pat.match(r["stem"])]
    print(f"  不符合『数字/字母-数字/字母[-3位序号]』的: {len(odd)} 张")
    for r in odd[:15]:
        print(f"    {r['rel_path']}")

    print(f"\n  体检表: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
