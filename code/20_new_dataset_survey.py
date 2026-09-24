"""只读勘察桌面上的新数据集 采样数据\\图片\\（用户要求：可查看，不得修改该文件夹）。

产出: outputs/新数据集勘察.csv 与 控制台汇总
"""
from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

# --- 统一路径（见 paths.py；数据搬家只改那里）---
sys.path.insert(0, str(Path(__file__).parent))
import paths as P  # noqa: E402

ROOT = P.RAW_D2
OUT = Path(r"D:\根系分割项目\outputs\新数据集勘察.csv")
Image.MAX_IMAGE_PIXELS = None


def main() -> int:
    files = sorted(p for p in ROOT.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".tif", ".tiff", ".png"))
    print(f"[扫描] {len(files)} 张")
    rows = []
    for i, p in enumerate(files, 1):
        rec = {"rel_path": str(p.relative_to(ROOT.parent)), "batch": p.parent.name,
               "name": p.name, "ext": p.suffix.lower(),
               "MB": round(p.stat().st_size / 1024 ** 2, 2)}
        try:
            with Image.open(p) as im:
                rec["w"], rec["h"] = im.size
                rec["mode"] = im.mode
                d = im.info.get("dpi")
                rec["dpi"] = f"{round(d[0])}x{round(d[1])}" if d else "无"
                rec["mm_w"] = round(im.width * 25.4 / (d[0] if d else 72), 1)
                rec["mm_h"] = round(im.height * 25.4 / (d[1] if d else 72), 1)
        except Exception as exc:  # noqa: BLE001
            rec.update({"w": -1, "h": -1, "mode": f"ERR:{type(exc).__name__}", "dpi": "?", "mm_w": 0, "mm_h": 0})
        rows.append(rec)
        if i % 40 == 0:
            print(f"  ... {i}/{len(files)}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    print("\n=== 按目录 ===")
    for b, c in sorted(Counter(r["batch"] for r in rows).items()):
        sub = [r for r in rows if r["batch"] == b]
        mb = sum(r["MB"] for r in sub)
        sizes = Counter(f'{r["w"]}x{r["h"]}' for r in sub)
        dpis = Counter(r["dpi"] for r in sub)
        exts = Counter(r["ext"] for r in sub)
        print(f"  {b:<8} {c:3d} 张  {mb:7.0f} MB  扩展名 {dict(exts)}")
        print(f"           dpi {dict(dpis)}")
        print(f"           尺寸 {dict(sizes.most_common(3))}")
    print(f"\n  总计 {len(rows)} 张  {sum(r['MB'] for r in rows)/1024:.2f} GB")
    print(f"  清单已写: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
