"""机器图残留清理 v1：按类型分别处理，避免误删断掉的细根。

三类残留，三种处理:
  A. 贴边黑块(扫描伪影)  = 碰到画幅边缘 **且** 绝大多数像素落在最外圈窄带里
                           -> 直接删（它不是根，是扫描仪盖板/黑边）
  B. 游离碎屑            = 不与主体连通 **且** 离主体很远
                           -> 删
  C. 贴近主体的小块      = 离主体近（<=KEEP_DIST）
                           -> 保留（很可能是被管线打断的细根，删了会更欠分割）

产出: outputs/残留分析/clean/ 下六张清理后的二值 TIFF + 前后对比 + 统计
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# --- 统一路径（见 paths.py；数据搬家只改那里）---
sys.path.insert(0, str(Path(__file__).parent))
import paths as P  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

Image.MAX_IMAGE_PIXELS = None
MAC = P.TEST_MAC
OUT = Path(r"D:\根系分割项目\outputs\残留分析")
CLEAN = OUT / "clean"
SAMPLES = ["28-CL-003", "126-003", "134-003", "66-003", "G218-001", "7-003"]

BIG = 2000          # 面积 >= 此值算主体
KEEP_DIST = 80      # 小块离主体 <= 此距离则保留（视为断根）
BAND_FRAC = 0.03    # 最外圈窄带宽度占画幅比例
BAND_IN = 0.80      # 贴边块有 >= 此比例的像素落在窄带内 -> 判为扫描伪影


def clean(a: np.ndarray):
    fg = a < 128
    H, W = fg.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), 8)
    band = np.zeros_like(fg)
    bh, bw = int(H * BAND_FRAC), int(W * BAND_FRAC)
    band[:bh, :] = band[-bh:, :] = band[:, :bw] = band[:, -bw:] = True

    drop = np.zeros_like(fg)
    stat = {"edge_removed": 0, "debris_removed": 0, "kept_near": 0, "edge_blocks": 0}
    # --- A. 贴边黑块 ---
    alive = []
    for j in range(1, n):
        x, y, w, h, area = st[j]
        touch = (x <= 0 or y <= 0 or x + w >= W or y + h >= H)
        if touch:
            m = lab == j
            if band[m].mean() >= BAND_IN:
                drop |= m
                stat["edge_removed"] += int(area)
                stat["edge_blocks"] += 1
                continue
        alive.append(j)

    # --- B/C. 游离碎屑 vs 断根 ---
    big = [j for j in alive if st[j, cv2.CC_STAT_AREA] >= BIG]
    main = np.isin(lab, big) & ~drop
    dist = cv2.distanceTransform((~main).astype(np.uint8), cv2.DIST_L2, 5)
    for j in alive:
        if st[j, cv2.CC_STAT_AREA] >= BIG:
            continue
        m = (lab == j) & ~drop
        if not m.any():
            continue
        if float(dist[m].min()) > KEEP_DIST:
            drop |= m
            stat["debris_removed"] += int(m.sum())
        else:
            stat["kept_near"] += int(m.sum())
    out = np.where(fg & ~drop, 0, 255).astype(np.uint8)
    stat["fg_before"] = int(fg.sum())
    stat["fg_after"] = int((fg & ~drop).sum())
    return out, drop, stat


def main() -> int:
    CLEAN.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"{'样本':<12}{'清理前px':>11}{'贴边块':>7}{'贴边删px':>10}"
          f"{'碎屑删px':>10}{'保留近块px':>11}{'清理后px':>11}{'减少':>8}")
    for s in SAMPLES:
        p = MAC / f"{s}-mac.tif"
        if not p.exists():
            continue
        a = np.asarray(Image.open(p).convert("L"))
        out, drop, st = clean(a)
        Image.fromarray(out).save(CLEAN / f"{s}-mac.tif", format="TIFF",
                                  compression="raw", dpi=(600, 600))
        red = 100 * (1 - st["fg_after"] / max(1, st["fg_before"]))
        print(f"{s:<12}{st['fg_before']:>11}{st['edge_blocks']:>7}{st['edge_removed']:>10}"
              f"{st['debris_removed']:>10}{st['kept_near']:>11}{st['fg_after']:>11}{red:>7.2f}%")
        rows.append({"样本": s, **st, "减少%": round(red, 2)})

    with (OUT / "清理统计.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    # 前后对比小图
    def th(p):
        im = Image.open(p).convert("L")
        a = np.asarray(im)
        k = max(1, int(np.ceil(max(a.shape) / 430)))
        Hc, Wc = (a.shape[0] // k) * k, (a.shape[1] // k) * k
        s = a[:Hc, :Wc].reshape(Hc // k, k, Wc // k, k).min(axis=(1, 3))
        return Image.fromarray(s.astype(np.uint8)).convert("RGB")

    CW, CH = 430, 310
    rowsn = len(rows)
    sheet = Image.new("RGB", (CW * 2, (CH + 26) * rowsn), (255, 255, 255))
    from PIL import ImageDraw, ImageFont
    try:
        f = ImageFont.truetype("msyh.ttc", 17)
    except Exception:  # noqa: BLE001
        f = ImageFont.load_default()
    dr = ImageDraw.Draw(sheet)
    for i, r in enumerate(rows):
        s = r["样本"]
        for c, (tag, path) in enumerate([("清理前", MAC / f"{s}-mac.tif"),
                                         ("清理后", CLEAN / f"{s}-mac.tif")]):
            if path.exists():
                t = th(path)
                t = t.resize((CW, int(t.height * CW / t.width)), Image.LANCZOS)
                sheet.paste(t, (c * CW, i * (CH + 26) + 26))
        dr.text((6, i * (CH + 26) + 4), f"{i+1}. {s}　　左=清理前　右=清理后"
                                       f"（贴边删 {r['edge_removed']}px，碎屑删 {r['debris_removed']}px，"
                                       f"保留近块 {r['kept_near']}px，总减 {r['减少%']}%）",
                fill=(0, 0, 0), font=f)
    sheet.save(OUT / "清理前后对比.png", optimize=True)
    print(f"\n  清理后 TIFF -> {CLEAN}")
    print(f"  对比图     -> {OUT/'清理前后对比.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
