"""全量机器图总览：分布检查 + 抽样预览图。"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
import paths as P  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

Image.MAX_IMAGE_PIXELS = None
ROOT = P.PROJ / "机器图"
OUT = P.OUTPUTS / "全量机器图"


def font(sz, b=False):
    for n in (("msyhbd.ttc", "msyh.ttc") if b else ("msyh.ttc",)):
        try:
            return ImageFont.truetype(n, sz)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


def thumb(p: Path, w: int, h: int) -> Image.Image:
    a = np.asarray(Image.open(p).convert("L"))
    k = max(1, int(np.ceil(max(a.shape) / max(w, h))))
    Hc, Wc = (a.shape[0] // k) * k, (a.shape[1] // k) * k
    s = a[:Hc, :Wc].reshape(Hc // k, k, Wc // k, k).min(axis=(1, 3)).astype(np.uint8)
    im = Image.fromarray(s).convert("RGB")
    k2 = min(w / im.width, h / im.height)
    im = im.resize((max(1, int(im.width * k2)), max(1, int(im.height * k2))), Image.LANCZOS)
    bg = Image.new("RGB", (w, h), (255, 255, 255))
    bg.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
    return bg


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader((ROOT / "_批量清单.csv").open(encoding="utf-8-sig")))
    print(f"清单 {len(rows)} 条")

    ds1 = [r for r in rows if r["输出"].startswith("数据集1")]
    ds2 = [r for r in rows if r["输出"].startswith("数据集2")]
    for tag, grp in [("数据集1", ds1), ("数据集2", ds2)]:
        v = np.array([r["根占比%"] for r in grp], dtype=float)
        print(f"\n{tag}（{len(grp)} 张）根占比: 中位 {np.median(v):.3f}%  "
              f"p05 {np.percentile(v,5):.3f}%  p95 {np.percentile(v,95):.3f}%  最大 {v.max():.3f}%")
        low = [r for r in grp if float(r["根占比%"]) < 0.3]
        if low:
            print(f"  根占比 <0.3% 的 {len(low)} 张（可能是空图或漏检，需人工看）:")
            for r in low[:10]:
                print(f"    {r['输出']:<46} {float(r['根占比%']):.3f}%")

    # 抽样预览：每个子目录抽 2 张
    subs = {}
    for r in rows:
        key = str(Path(r["输出"]).parent)
        subs.setdefault(key, []).append(r)
    picks = []
    for key in sorted(subs):
        picks += subs[key][:2]
    CW, CH = 380, 280
    cols = 4
    rws = (len(picks) + cols - 1) // cols
    sheet = Image.new("RGB", (CW * cols, (CH + 24) * rws), (255, 255, 255))
    dr = ImageDraw.Draw(sheet)
    for i, r in enumerate(picks):
        c, rr = i % cols, i // cols
        x, y = c * CW, rr * (CH + 24)
        p = ROOT / r["输出"]
        if p.exists():
            sheet.paste(thumb(p, CW - 6, CH), (x + 3, y + 22))
        dr.text((x + 4, y + 2), f"{Path(r['输出']).name}  {r['根占比%']}%",
                fill=(0, 0, 0), font=font(15))
    o = OUT / "全量机器图_抽样预览.png"
    sheet.save(o, optimize=True)
    print(f"\n  抽样预览 -> {o}  {sheet.size}")

    # 重采样统计
    res = [r for r in rows if r.get("重采样") == "True"]
    if res:
        print(f"\n  非 600dpi 被重采样的 {len(res)} 张:")
        for r in res[:12]:
            print(f"    {r['输出']:<46} 源 {r['源dpi']}dpi -> {r['尺寸']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
