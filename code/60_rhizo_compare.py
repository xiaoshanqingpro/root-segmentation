"""列出 WinRHIZO 全部记录的关键指标，并做 人工/CV/U-Net 三方对比。

列索引（按表头精确对齐，见 59_parse_rhizo.py 输出）:
  0 SampleId, 3 ImageFileName, 5 AnalysedRegionArea,
  15 Length(cm), 17 ProjArea(cm2), 19 SurfArea(cm2), 21 AvgDiam(mm),
  25 RootVolume(cm3), 28 Tips, 29 Forks, 30 Crossings, 31 NofLinks
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

SRC = Path.home() / "Desktop" / "测试" / "测试数据.TXT"
OUT = Path(r"D:\根系分割项目\outputs\Rhizo三方对比")
OUT.mkdir(parents=True, exist_ok=True)

IDX = {"Length": 15, "ProjArea": 17, "SurfArea": 19, "AvgDiam": 21,
       "Volume": 25, "Tips": 28, "Forks": 29, "Crossings": 30, "RegionArea": 5}


def read_rows(path: Path):
    raw = path.read_bytes()
    for enc in ("gbk", "utf-8-sig", "utf-8", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    lines = [ln for ln in text.splitlines() if ln.strip()]
    hdr_i = next(i for i, ln in enumerate(lines) if ln.startswith("RHIZO"))
    recs = []
    for ln in lines[hdr_i + 1:]:
        c = ln.split("\t")
        if not c or not c[0] or c[0].startswith("SampleId"):
            continue
        recs.append(c)
    return recs


def classify(sid: str):
    """SampleId -> (样本, 版本)"""
    s = sid.lower()
    if "unet" in s:
        ver = "U-Net"
    elif s.endswith("-man") or "-man-" in s:
        ver = "人工"
    elif "mac" in s:
        ver = "机器mac"
    else:
        ver = "?"
    m = re.match(r"^(.+?)[-_](?:man|mac|unet)", sid, re.I)
    stem = m.group(1) if m else sid
    stem = stem.replace("-MAN", "").replace("-MAC", "")
    return stem, ver


def f(v: str):
    try:
        return float(v)
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    recs = read_rows(SRC)
    rows = []
    for c in recs:
        sid = c[0].strip()
        stem, ver = classify(sid)
        r = {"SampleId": sid, "样本": stem, "版本": ver,
             "Image": c[3].strip() if len(c) > 3 else ""}
        for k, i in IDX.items():
            r[k] = f(c[i]) if i < len(c) else None
        rows.append(r)

    rows.sort(key=lambda r: (r["样本"], r["版本"]))
    print(f"{'SampleId':<18}{'样本':<12}{'版本':<8}{'Length':>10}{'SurfArea':>10}"
          f"{'AvgDiam':>9}{'Volume':>9}{'Tips':>7}{'Forks':>7}{'Xing':>6}   Image")
    for r in rows:
        def g(k, d=3):
            v = r[k]
            return f"{v:.{d}f}" if isinstance(v, float) else "—"
        print(f"{r['SampleId']:<18}{r['样本']:<12}{r['版本']:<8}"
              f"{g('Length',2):>10}{g('SurfArea',3):>10}{g('AvgDiam',4):>9}"
              f"{g('Volume',3):>9}{g('Tips',0):>7}{g('Forks',0):>7}{g('Crossings',0):>6}"
              f"   {r['Image']}")

    with (OUT / "rhizo_all_records.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"\n  全部记录 -> {OUT/'rhizo_all_records.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
