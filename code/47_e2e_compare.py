"""汇总 6 张样本的「人工修图 vs 机器修图」WinRHIZO 端到端测量对比。

数据源: C:\\Users\\<用户名>\\Desktop\\测试\\测试数据.TXT（14 条记录）
产出: outputs/端到端对比/端到端指标.csv + 对比图 + 控制台表
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# --- 统一路径（见 paths.py；数据搬家只改那里）---
sys.path.insert(0, str(Path(__file__).parent))
import paths as P  # noqa: E402

# Windows 控制台默认 GBK，输出 ² ³ 这类字符会崩；强制 UTF-8 并容错
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

TXT = P.TEST_DATA
OUT = Path(r"D:\根系分割项目\outputs\端到端对比")

WANT = ["Length(cm)", "ProjArea(cm2)", "SurfArea(cm2)", "AvgDiam(mm)",
        "RootVolume(cm3)", "Tips", "Forks", "Crossings", "AnalysedRegionArea(cm2)"]
CN = {"Length(cm)": "根总长(cm)", "ProjArea(cm2)": "投影面积(cm²)", "SurfArea(cm2)": "表面积(cm²)",
      "AvgDiam(mm)": "平均直径(mm)", "RootVolume(cm3)": "根体积(cm³)", "Tips": "根尖数",
      "Forks": "分叉数", "Crossings": "交叉数", "AnalysedRegionArea(cm2)": "分析区面积(cm²)"}
KEY = ["Length(cm)", "SurfArea(cm2)", "RootVolume(cm3)", "AvgDiam(mm)"]

# 6 个样本: 人工记录名 -> 机器记录名
PAIRS = [("28-CL-003-man", "28-CL-003-mac"), ("66-003-man", "66-003-mac"),
         ("126-003-man", "126-003-mac"), ("134-003-man", "134-003-mac"),
         ("G218-001-man", "G218-001-MAC"), ("7003man", "7003auto2-tiff")]
SAMPLE_NAME = {"28-CL-003-man": "28-CL-003", "66-003-man": "66-003", "126-003-man": "126-003",
               "134-003-man": "134-003", "G218-001-man": "G218-001", "7003man": "7-003"}


def font(sz, b=False):
    for n in (("msyhbd.ttc", "msyh.ttc") if b else ("msyh.ttc",)):
        try:
            return ImageFont.truetype(n, sz)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


def parse():
    txt = TXT.read_text(encoding="gbk", errors="replace")
    lines = [l.rstrip("\n") for l in txt.splitlines()]
    names = [h.strip() for h in next(l for l in lines if "SoilVol(m3)" in l).split("\t")]
    recs = {}
    for l in lines:
        f = l.split("\t")
        if len(f) < 5:
            continue
        sid = f[0].strip()
        if sid in ("SampleId", "RHIZO 2016a", ""):
            continue
        row = {}
        for w in WANT:
            try:
                row[w] = float(f[names.index(w)])
            except Exception:  # noqa: BLE001
                row[w] = float("nan")
        recs[sid] = row          # 同名的后来者覆盖前者
    return recs


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    recs = parse()
    rows = []
    for man_id, mac_id in PAIRS:
        if man_id not in recs or mac_id not in recs:
            print(f"  缺记录: {man_id} / {mac_id}")
            continue
        m, k = recs[man_id], recs[mac_id]
        r = {"样本": SAMPLE_NAME[man_id]}
        for w in WANT:
            r[f"人工_{CN[w]}"] = round(m[w], 4)
            r[f"机器_{CN[w]}"] = round(k[w], 4)
            r[f"差异%_{CN[w]}"] = round(100 * (k[w] / m[w] - 1), 1) if m[w] else float("nan")
        rows.append(r)

    keys = list(rows[0].keys())
    with (OUT / "端到端指标.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)

    # ---- 控制台表 ----
    print("=== 四项重点指标: 机器 vs 人工 差异% ===")
    print(f"{'样本':<12}" + "".join(f"{CN[w]:>14}" for w in KEY))
    for r in rows:
        print(f"{r['样本']:<12}" + "".join(f"{r['差异%_'+CN[w]]:>13.1f}%" for w in KEY))
    print()
    for w in KEY:
        v = np.array([r[f"差异%_{CN[w]}"] for r in rows], dtype=float)
        print(f"  {CN[w]:<14} 中位 {np.median(v):+7.1f}%   均值 {v.mean():+7.1f}%   "
              f"范围 [{v.min():+.1f}%, {v.max():+.1f}%]   最大绝对偏差 {np.abs(v).max():.1f}%")

    # ---- 对比图 ----
    CW, RH = 420, 260
    fig = Image.new("RGB", (CW * 4 + 40, RH * len(rows) + 120), (255, 255, 255))
    dr = ImageDraw.Draw(fig)
    dr.text((14, 10), "6 张样本端到端测量对比（同一台扫描仪 / 同一套阈值）",
            fill=(0, 0, 0), font=font(26, True))
    for j, w in enumerate(KEY):
        dr.text((20 + j * CW, 52), CN[w], fill=(0, 0, 0), font=font(22, True))
    y = 92
    for r in rows:
        dr.text((20, y + RH // 2 - 12), r["样本"], fill=(0, 0, 0), font=font(22, True))
        for j, w in enumerate(KEY):
            a, b = r[f"人工_{CN[w]}"], r[f"机器_{CN[w]}"]
            d = r[f"差异%_{CN[w]}"]
            x = 20 + j * CW
            box = [x + 120, y + 20, x + CW - 10, y + RH - 20]
            dr.rectangle(box, outline=(210, 210, 210))
            mx = max(a, b) * 1.15
            ha = (box[3] - box[1]) * a / mx
            hb = (box[3] - box[1]) * b / mx
            dr.rectangle([box[0], box[3] - ha, box[0] + 45, box[3]], fill=(90, 90, 90))
            dr.rectangle([box[0] + 55, box[3] - hb, box[0] + 100, box[3]], fill=(0, 150, 200))
            dr.text((box[0] + 2, box[3] - ha - 20), f"{a:g}", fill=(0, 0, 0), font=font(15))
            dr.text((box[0] + 57, box[3] - hb - 20), f"{b:g}", fill=(0, 0, 0), font=font(15))
            col = (200, 0, 0) if abs(d) > 20 else ((190, 120, 0) if abs(d) > 10 else (0, 130, 0))
            dr.text((x + 130, y + RH - 16), f"{d:+.1f}%", fill=col, font=font(19, True))
        y += RH
    dr.text((20, y + 6), "■ 人工修图    ■ 机器修图     差异 = (机器-人工)/人工",
            fill=(70, 70, 70), font=font(18))
    fig.save(OUT / "端到端对比图.png", optimize=True)
    print(f"\n  -> {OUT/'端到端指标.csv'}")
    print(f"  -> {OUT/'端到端对比图.png'}")
    return 0


if __name__ == "__main__":
    main()
