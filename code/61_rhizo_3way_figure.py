"""人工 / CV(机器) / U-Net 三方对比图。

按 **ImageFileName** 识别样本与版本（SampleId 有错标：218-003-unet 实为 281-003_UNet.tif）。

产出: outputs/Rhizo三方对比/三方对比.png
        outputs/Rhizo三方对比/对比数据.csv
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import numpy as np

SRC = Path.home() / "Desktop" / "测试" / "测试数据.TXT"
OUT = Path(r"D:\根系分割项目\outputs\Rhizo三方对比")
OUT.mkdir(parents=True, exist_ok=True)

# Windows 控制台默认 GBK，打不出 ² ³ 等字符
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

IDX = {"Length(cm)": 15, "ProjArea(cm2)": 17, "SurfArea(cm2)": 19,
       "AvgDiam(mm)": 21, "Volume(cm3)": 25, "Tips": 28, "Forks": 29, "Crossings": 30}
CN = {"Length(cm)": "根总长(cm)", "ProjArea(cm2)": "投影面积(cm²)", "SurfArea(cm2)": "表面积(cm²)",
      "AvgDiam(mm)": "平均直径(mm)", "Volume(cm3)": "根体积(cm³)", "Tips": "根尖数",
      "Forks": "分叉数", "Crossings": "交叉数"}
SAMPLES = ["7-003", "G218-001", "281-003"]


def read_rows(path: Path):
    raw = path.read_bytes()
    text = None
    for enc in ("gbk", "utf-8-sig", "utf-8", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    lines = [ln for ln in text.splitlines() if ln.strip()]
    hi = next(i for i, ln in enumerate(lines) if ln.startswith("RHIZO"))
    return [ln.split("\t") for ln in lines[hi + 1:]
            if ln.split("\t")[0].strip() and not ln.startswith("SampleId")]


def classify(img: str, sid: str):
    """按 ImageFileName 判样本与版本"""
    n = img.lower()
    if "unet" in n or "unet" in sid.lower():
        ver = "U-Net"
    elif "-man" in n or n.endswith("man.tif"):
        ver = "人工"
    elif "-mac" in n or "_a_" in n:
        ver = "CV"
    else:
        return None, None
    m = re.match(r"^([0-9A-Za-z\-]+?)[-_](?:man|mac|unet|MAN|MAC|UNet)", img)
    stem = m.group(1) if m else None
    if stem and stem.upper() == "G218-001":
        stem = "G218-001"
    return stem, ver


def f(v):
    try:
        return float(v)
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    data: dict[str, dict[str, dict]] = {}
    for c in read_rows(SRC):
        sid = c[0].strip()
        img = (c[3].strip() if len(c) > 3 else "")
        stem, ver = classify(img, sid)
        if stem is None:
            continue
        rec = {k: (f(c[i]) if i < len(c) else None) for k, i in IDX.items()}
        rec["_img"], rec["_sid"] = img, sid
        data.setdefault(stem, {})[ver] = rec

    print(f"{'样本':<12}{'版本':<8}" + "".join(f"{CN[k]:>13}" for k in IDX))
    rows = []
    for s in SAMPLES:
        for ver in ("人工", "CV", "U-Net"):
            r = data.get(s, {}).get(ver)
            if not r:
                print(f"{s:<12}{ver:<8}  缺")
                continue
            print(f"{s:<12}{ver:<8}" + "".join(
                f"{(r[k] if r[k] is not None else float('nan')):>13.4g}" for k in IDX))
            rows.append({"样本": s, "版本": ver, **{k: r[k] for k in IDX}, "图": r["_img"]})

    with (OUT / "对比数据.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    # ---------------- 画图 ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager, rcParams
    _fp = None
    for fn in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\msyhbd.ttc",
               r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\simsun.ttc"):
        if Path(fn).exists():
            try:
                font_manager.fontManager.addfont(fn)
                _fp = font_manager.FontProperties(fname=fn)
                break
            except Exception:  # noqa: BLE001
                continue
    if _fp is None:
        print("  !! 未找到中文字体，图上可能显示方框")
    else:
        rcParams["font.family"] = _fp.get_name()
        print(f"  中文字体: {_fp.get_name()}  ({_fp.get_file()})")
    rcParams["axes.unicode_minus"] = False

    metrics = ["Length(cm)", "SurfArea(cm2)", "AvgDiam(mm)", "Volume(cm3)"]
    colors = {"人工": "#c0392b", "CV": "#2c5fa8", "U-Net": "#1e8449"}
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.5, 1, 1], hspace=0.45, wspace=0.28)

    # 上：数据表
    ax = fig.add_subplot(gs[0, :])
    ax.axis("off")
    ax.set_title("WinRHIZO 扫描结果：人工 / CV / U-Net 三方对比（同一张扫描件、同一套阈值）",
                 fontsize=15, pad=12)
    cols = ["样本", "版本", "根总长(cm)", "表面积(cm²)", "平均直径(mm)", "根体积(cm³)",
            "根尖数", "分叉数", "交叉数", "根总长 vs 人工"]
    cell = []
    for s in SAMPLES:
        base = data.get(s, {}).get("人工", {}).get("Length(cm)")
        for ver in ("人工", "CV", "U-Net"):
            r = data.get(s, {}).get(ver)
            if not r:
                continue
            d = "—"
            if base and r["Length(cm)"]:
                d = f"{(r['Length(cm)']-base)/base*100:+.1f}%"
            cell.append([s, ver,
                         f"{r['Length(cm)']:.1f}", f"{r['SurfArea(cm2)']:.2f}",
                         f"{r['AvgDiam(mm)']:.4f}", f"{r['Volume(cm3)']:.3f}",
                         f"{r['Tips']:.0f}", f"{r['Forks']:.0f}", f"{r['Crossings']:.0f}", d])
    tb = ax.table(cellText=cell, colLabels=cols, loc="center", cellLoc="center")
    tb.auto_set_font_size(False)
    tb.set_fontsize(10.5)
    tb.scale(1, 1.55)
    for j in range(len(cols)):
        tb[0, j].set_facecolor("#34495e")
        tb[0, j].set_text_props(color="white", weight="bold")
    for i in range(1, len(cell) + 1):
        tb[i, 1].set_text_props(color=colors.get(cell[i - 1][1], "black"), weight="bold")
        if cell[i - 1][1] == "人工":
            for j in range(len(cols)):
                tb[i, j].set_facecolor("#fdf0ef")

    # 中：绝对量柱状图
    for k, metric in enumerate(metrics):
        ax = fig.add_subplot(gs[1, k])
        x = np.arange(len(SAMPLES))
        w = 0.26
        for oi, ver in enumerate(("人工", "CV", "U-Net")):
            vals = [data.get(s, {}).get(ver, {}).get(metric, np.nan) for s in SAMPLES]
            ax.bar(x + (oi - 1) * w, vals, w, label=ver, color=colors[ver])
        ax.set_xticks(x)
        ax.set_xticklabels(SAMPLES, fontsize=9)
        ax.set_title(CN[metric], fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        if k == 0:
            ax.legend(fontsize=9)

    # 下：相对人工的偏差
    for k, metric in enumerate(metrics):
        ax = fig.add_subplot(gs[2, k])
        x = np.arange(len(SAMPLES))
        w = 0.32
        for oi, ver in enumerate(("CV", "U-Net")):
            vals = []
            for s in SAMPLES:
                b = data.get(s, {}).get("人工", {}).get(metric)
                v = data.get(s, {}).get(ver, {}).get(metric)
                vals.append((v - b) / b * 100 if (b and v) else np.nan)
            bars = ax.bar(x + (oi - 0.5) * w, vals, w, label=f"{ver} vs 人工", color=colors[ver])
            for bb, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(bb.get_x() + bb.get_width() / 2, v, f"{v:+.0f}%",
                            ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
        ax.axhline(0, color="k", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(SAMPLES, fontsize=9)
        ax.set_title(f"{CN[metric]} 相对人工的偏差", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylabel("%", fontsize=9)
        if k == 0:
            ax.legend(fontsize=9)

    o = OUT / "三方对比.png"
    fig.savefig(o, dpi=115, bbox_inches="tight")
    print(f"\n  -> {o}")
    print(f"  -> {OUT/'对比数据.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
