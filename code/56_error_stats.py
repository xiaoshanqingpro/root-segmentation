"""解析 测试数据.TXT 的全部记录，理清 man/mac 配对，并算端到端误差（含置信区间）。

产出:
  outputs/误差分析/records.csv      全部原始记录
  outputs/误差分析/pairs.csv        man/mac 配对 + 逐项误差
  outputs/误差分析/summary.json     误差统计 + bootstrap 95% CI
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

PROJ = Path(r"D:\根系分割项目")
TXT = Path(r"C:\Users\LIHAOYANG\Desktop\测试\测试数据.TXT")
OUT = PROJ / "outputs" / "误差分析"

METRICS = [("Length(cm)", "根总长", "cm"), ("SurfArea(cm2)", "表面积", "cm²"),
           ("RootVolume(cm3)", "根体积", "cm³"), ("AvgDiam(mm)", "平均直径", "mm"),
           ("ProjArea(cm2)", "投影面积", "cm²"), ("Tips", "根尖数", ""),
           ("Forks", "分叉数", ""), ("Crossings", "交叉数", "")]


def parse():
    txt = TXT.read_text(encoding="gbk", errors="replace")
    lines = [l.rstrip("\n") for l in txt.splitlines()]
    hdr = next(l for l in lines if "SoilVol(m3)" in l)
    names = [h.strip() for h in hdr.split("\t")]
    recs = []
    for l in lines:
        f = l.split("\t")
        if len(f) < 6 or f[0].strip() in ("SampleId", "RHIZO 2016a", ""):
            continue
        rec = {"id": f[0].strip(), "image": f[3].strip() if len(f) > 3 else ""}
        for k, _, _ in METRICS:
            try:
                rec[k] = float(f[names.index(k)])
            except Exception:  # noqa: BLE001
                rec[k] = float("nan")
        recs.append(rec)
    return recs


def norm(s: str) -> str:
    """把记录 id / 文件名归一到样本号"""
    s = s.strip()
    for suf in ("-man-9.22", "-mac-9.22", "_man", "_mac"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    s = s.replace("-man", "").replace("-mac", "").replace("-MAC", "")
    s = s.replace(".tif", "").replace(".jpg", "")
    s = s.replace("7003auto2-tiff", "7-003").replace("7003auto", "7-003")
    s = s.replace("7003man", "7-003")
    s = s.replace("00000-1", "00000-001")
    return s


def boot_ci(x: np.ndarray, n: int = 10000, seed: int = 42) -> tuple[float, float]:
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    meds = np.array([np.median(rng.choice(x, len(x), replace=True)) for _ in range(n)])
    return (float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5)))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    recs = parse()
    print(f"原始记录 {len(recs)} 条")

    with (OUT / "records.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(recs[0].keys()), extrasaction="ignore")
        wr.writeheader()
        wr.writerows(recs)

    man, mac = {}, {}
    for r in recs:
        s = norm(r["id"])
        # 优先用文件名的 -man / -mac 判断
        img = r["image"].lower()
        if "-man" in img:
            man[s] = r
        elif "-mac" in img:
            mac[s] = r
        elif r["id"].endswith("-man"):
            man[s] = r
        elif r["id"].lower().endswith("-mac"):
            mac[s] = r
    print(f"人工记录 {len(man)} 个样本: {sorted(man)}")
    print(f"机器记录 {len(mac)} 个样本: {sorted(mac)}")
    common = sorted(set(man) & set(mac))
    print(f"可配对 {len(common)} 对: {common}")
    only_m = sorted(set(man) - set(mac))
    only_k = sorted(set(mac) - set(man))
    if only_m:
        print(f"  只有人工: {only_m}")
    if only_k:
        print(f"  只有机器: {only_k}")

    rows = []
    for s in common:
        row = {"样本": s}
        for k, cn, _ in METRICS:
            a, b = man[s].get(k, float("nan")), mac[s].get(k, float("nan"))
            row[f"人工_{cn}"] = round(a, 4) if np.isfinite(a) else ""
            row[f"机器_{cn}"] = round(b, 4) if np.isfinite(b) else ""
            row[f"误差%_{cn}"] = round(100 * (b / a - 1), 2) if a else ""
        rows.append(row)
    with (OUT / "pairs.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    print(f"\n=== 端到端误差（机器 vs 人工），n={len(common)} ===")
    print(f"{'指标':<10}{'中位误差':>10}{'95% CI':>22}{'均值':>9}{'绝对中位':>10}{'范围':>22}")
    summary = {}
    for k, cn, _ in METRICS:
        v = np.array([100 * (mac[s][k] / man[s][k] - 1) for s in common
                      if man[s].get(k) and mac[s].get(k)], dtype=float)
        v = v[np.isfinite(v)]
        if not len(v):
            continue
        lo, hi = boot_ci(v)
        summary[cn] = {"n": int(len(v)), "median": round(float(np.median(v)), 2),
                       "ci95": [round(lo, 2), round(hi, 2)],
                       "mean": round(float(v.mean()), 2),
                       "median_abs": round(float(np.median(np.abs(v))), 2),
                       "min": round(float(v.min()), 2), "max": round(float(v.max()), 2),
                       "significant_from_0": bool(lo > 0 or hi < 0)}
        ss = summary[cn]
        sig = "显著" if ss["significant_from_0"] else "不显著"
        rng_s = "[{:.1f}, {:.1f}]".format(ss["min"], ss["max"])
        ci_s = "[{:.2f}, {:.2f}]".format(lo, hi)
        print(f"{cn:<10}{ss['median']:>9.2f}%{ci_s:>22}{ss['mean']:>8.2f}%"
              f"{ss['median_abs']:>9.2f}%{rng_s:>22}  {sig}")

    (OUT / "summary.json").write_text(json.dumps(
        {"n_pairs": len(common), "samples": common, "metrics": summary},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
