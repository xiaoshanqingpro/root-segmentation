"""量化 EDGE_KILL（贴边伪影剔除）规则的影响面。

对每个数据集抽样，比较 EDGE_KILL 开/关两种情况下判出的根占比，
找出"关掉后根显著变多"的图 —— 这些就是根被误杀的受害者。

用法:
    python 67_edgekill_impact.py [每组抽样数]
输出:
    outputs\\edgekill_影响_<时间戳>.csv
"""
from __future__ import annotations

import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import rootseg_common as R  # noqa: E402
import paths as P  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

Image.MAX_IMAGE_PIXELS = None
EXTS = (".jpg", ".jpeg", ".tif", ".tiff", ".png")
SHANKUO = Path.home() / "Desktop" / "杉阔混交林"

SETS = {
    "数据集1": P.RAW_D1,
    "数据集2": P.RAW_D2,
    "杉阔-F": SHANKUO / "F",
    "杉阔-F补": SHANKUO / "F补",
    "杉阔-9.8校准": SHANKUO / "9.8校准",
}


def set_params():
    R.FEAT_LONG = 3000
    R.HALO_GROW = 0
    R.SHRINK_PX = 1
    R.SMOOTH_ALL = True
    R.SMOOTH_OPEN_PX = 1
    R.BRIDGE_CLOSE_PX = 2


def load(p: Path) -> np.ndarray:
    im = Image.open(p)
    dpi = float((im.info.get("dpi") or (600, 600))[0])
    rgb = np.asarray(im.convert("RGB"))
    if abs(dpi - 600.0) > 1:
        k = 600.0 / dpi
        rgb = np.asarray(Image.fromarray(rgb).resize(
            (max(1, round(rgb.shape[1] * k)), max(1, round(rgb.shape[0] * k))), Image.LANCZOS))
    return rgb


def main(n_sample: int = 40, seed: int = 20260924) -> int:
    set_params()
    random.seed(seed)
    rows = []
    t0 = time.time()
    for name, root_dir in SETS.items():
        if not root_dir.exists():
            print(f"  跳过 {name}（不存在）")
            continue
        files = sorted(p for p in root_dir.rglob("*") if p.suffix.lower() in EXTS)
        # 数据集2 里含重复的 9.15，抽样时无妨
        take = files if len(files) <= n_sample else random.sample(files, n_sample)
        print(f"\n[{name}] {len(files)} 张中抽 {len(take)} 张")
        for p in take:
            rgb = load(p)
            R.EDGE_KILL = True
            r_on, _, info_on = R.extract_root(rgb)
            R.EDGE_KILL = False
            r_off, _, info_off = R.extract_root(rgb)
            on, off = info_on["root_pct"], info_off["root_pct"]
            gain = off - on
            rows.append({"组": name, "文件": p.name, "原图": f"{rgb.shape[1]}x{rgb.shape[0]}",
                         "关规则根占比%": off, "开规则根占比%": on, "差值": round(gain, 4),
                         "倍数": round(off / on, 2) if on > 0 else float("inf"),
                         "前景%": info_on["fg_pct"]})
            flag = "  <<< 被误杀" if gain > max(0.05, 0.5 * off) else ""
            print(f"    {p.name:<24} 关 {off:7.4f}%  开 {on:7.4f}%  差 {gain:7.4f}{flag}", flush=True)
    R.EDGE_KILL = True

    out = P.PROJ / "outputs" / f"edgekill_影响_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)

    print("\n" + "=" * 66)
    print(f"{'组':<14}{'张数':>5}{'受影响':>8}{'占比':>8}{'最大倍数':>10}")
    for name in SETS:
        sub = [r for r in rows if r["组"] == name]
        if not sub:
            continue
        bad = [r for r in sub if r["差值"] > 0.05 and r["关规则根占比%"] > 1.3 * r["开规则根占比%"]]
        mx = max((r["倍数"] for r in sub if r["倍数"] != float("inf")), default=0)
        print(f"{name:<14}{len(sub):>5}{len(bad):>8}{len(bad)/len(sub)*100:>7.0f}%{mx:>10.2f}")
    print(f"\n清单 -> {out}   用时 {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 40))
