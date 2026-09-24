"""解析 WinRHIZO 输出（测试数据.TXT，GBK + 制表符），按表头精确对齐列。"""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path.home() / "Desktop" / "测试" / "测试数据.TXT"

WANT = ["Length(cm)", "ProjArea(cm2)", "SurfArea(cm2)", "AvgDiam(mm)",
        "RootVolume(cm3)", "Tips", "Forks", "Crossings",
        "AnalysedRegionArea(cm2)", "NofLinks"]


def read_rows(path: Path):
    raw = path.read_bytes()
    for enc in ("gbk", "utf-8-sig", "utf-8", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    lines = [ln for ln in text.splitlines() if ln.strip()]
    # 表头 = 以 RHIZO 开头的那行
    hdr_i = next(i for i, ln in enumerate(lines) if ln.startswith("RHIZO"))
    hdr = lines[hdr_i].split("\t")
    recs = []
    for ln in lines[hdr_i + 1:]:
        c = ln.split("\t")
        if not c or not c[0] or c[0].startswith("SampleId"):
            continue
        rec = {}
        for i, h in enumerate(hdr):
            rec[h.strip()] = c[i].strip() if i < len(c) else ""
        rec["_col0"] = c[0].strip()
        rec["_cols"] = len(c)
        recs.append(rec)
    return hdr, recs


def main() -> int:
    hdr, recs = read_rows(SRC)
    print(f"表头 {len(hdr)} 列，数据 {len(recs)} 行\n")
    print("=== 表头逐列（便于核对索引）===")
    for i, h in enumerate(hdr):
        print(f"  {i:2d}  {h}")
    print("\n=== 第一行数据的键值 ===")
    if recs:
        r = recs[0]
        for i, h in enumerate(hdr):
            print(f"  {h.strip():<44} = {r.get(h.strip(), '')!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
