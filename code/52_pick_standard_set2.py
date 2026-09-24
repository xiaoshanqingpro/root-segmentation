"""抽取第二批典型图（5 张），用于扩人工真值。

与第一批的区别:
  - 排除已用的 6 张（标准集 5 张 + 7-003）
  - 特征维度加两项: 细根占比（半宽 <=2px 的像素占比）、贴边伪影强度
    —— 因为当前最大短板是"极淡细根"，需要有代表它、也要有代表其他类型的

产出: outputs/标准图集2/标准图集2_contact.png + csv
"""
from __future__ import annotations

import csv
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
import rootseg_common as R  # noqa: E402

# --- 统一路径（见 paths.py；数据搬家只改那里）---
sys.path.insert(0, str(Path(__file__).parent))
import paths as P  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

Image.MAX_IMAGE_PIXELS = None
PROJ = P.PROJ
SRC2 = P.RAW_D2
OUT = PROJ / "outputs" / "标准图集2"
N_PICK = 5
WORKERS = 8
DONE = {"7-001", "7-002", "7-003", "66-003", "28-CL-003", "G218-001", "126-003", "134-003"}


def features(src: str):
    cv2.setNumThreads(1)
    p = Path(src)
    rgb = np.asarray(Image.open(p).convert("RGB"))
    H, W = rgb.shape[:2]
    sc = min(1.0, 1400 / max(H, W))
    small = cv2.resize(rgb, (int(W * sc), int(H * sc)), interpolation=cv2.INTER_AREA) if sc < 1 else rgb
    hh, ww = small.shape[:2]
    fg = ~R.flood_background(small)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1].astype(np.float32) / 255.0
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32)
    k = max(31, (int(min(hh, ww) * 0.05) // 2) * 2 + 1)
    illum = cv2.morphologyEx(gray.astype(np.uint8), cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(np.float32)
    depth = illum - gray
    deep = depth > R.DARK_DROP
    core = fg & deep & ((sat >= R.SAT_ROOT) | (gray < R.GRAY_TUB))
    if core.sum() < 50:
        core = fg & deep
    dt = cv2.distanceTransform(core.astype(np.uint8), cv2.DIST_L2, 5)
    half = dt[core] / max(sc, 1e-6)
    # 贴边伪影强度: 最外圈 3% 带内的前景占比
    bh, bw = int(hh * 0.03), int(ww * 0.03)
    band = np.zeros_like(fg)
    band[:bh, :] = band[-bh:, :] = band[:, :bw] = band[:, -bw:] = True
    edge_int = 100 * float((fg & band).sum()) / max(1, int(band.sum()))
    return {"src": src,
            "rel": str(p.relative_to(SRC2)),
            "root_gray": round(float(np.median(gray[core])), 1),
            "root_sat": round(float(np.median(sat[core])), 4),
            "root_width_px": round(float(np.median(half)) * 2, 2),
            "fine_pct": round(100 * float((half <= 1.0).mean()), 2),   # 细根占比
            "shadow_pct": round(100 * float((fg & deep & ~core).mean()), 3),
            "fg_pct": round(100 * float(fg.mean()), 3),
            "root_pct": round(100 * float(core.mean()), 3),
            "edge_artifact": round(edge_int, 2)}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    files = []
    for p in sorted(SRC2.rglob("*")):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".tif", ".tiff", ".png"):
            continue
        rel = p.relative_to(SRC2)
        if any(x in rel.parts for x in ("9.9改", "work", "ocr_raw")) or p.stem in DONE:
            continue
        files.append(str(p))
    print(f"[标准图集2] 候选 {len(files)} 张（已排除已用的 8 张）")
    rows, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for fu in as_completed([ex.submit(features, f) for f in files]):
            try:
                rows.append(fu.result())
            except Exception:  # noqa: BLE001
                pass
    print(f"  特征完成 {len(rows)} 张，{time.time()-t0:.0f}s")

    # 过滤: 排除"内容太少"的图（前景 <1.5%，或根占比 <0.5%）
    # 理由: 几乎是空白的图，人工精修性价比极低，训练信息量也小
    before = len(rows)
    rows = [r for r in rows if r["root_pct"] >= 0.8]
    print(f"  过滤掉根内容过少的 {before - len(rows)} 张，剩 {len(rows)} 张候选")

    keys = ["root_gray", "root_sat", "root_width_px", "fine_pct", "shadow_pct", "fg_pct",
            "edge_artifact"]
    X = np.array([[r[k] for k in keys] for r in rows], dtype=float)
    Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    sel = [int(np.argmax(np.linalg.norm(Z - Z.mean(0), axis=1)))]
    for _ in range(N_PICK - 1):
        d = np.min(np.linalg.norm(Z[:, None, :] - Z[sel][None, :, :], axis=2), axis=1)
        d[sel] = -1
        sel.append(int(np.argmax(d)))
    picked = [rows[i] for i in sel]

    with (OUT / "标准图集2.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    print(f"\n  选出 {len(picked)} 张:")
    for r in picked:
        print(f"    {r['rel']:<36} 灰度 {r['root_gray']:6.1f}  饱和 {r['root_sat']:.3f}  "
              f"宽 {r['root_width_px']:5.2f}px  根占比 {r['root_pct']:5.2f}%  "
              f"阴影 {r['shadow_pct']:6.2f}%  贴边 {r['edge_artifact']:5.2f}%")

    def font(sz, b=False):
        for n in (("msyhbd.ttc", "msyh.ttc") if b else ("msyh.ttc",)):
            try:
                return ImageFont.truetype(n, sz)
            except Exception:  # noqa: BLE001
                continue
        return ImageFont.load_default()

    CW, CH = 700, 500
    tiles = []
    for r in picked:
        im = Image.open(r["src"]).convert("RGB")
        a = np.asarray(im)
        k = max(1, int(np.ceil(max(a.shape[:2]) / max(CW, CH))))
        Hc, Wc = (a.shape[0] // k) * k, (a.shape[1] // k) * k
        s = a[:Hc, :Wc].reshape(Hc // k, k, Wc // k, k, 3).min(axis=(1, 3)).astype(np.uint8)
        t = Image.fromarray(s)
        k2 = min(CW / t.width, CH / t.height)
        t = t.resize((int(t.width * k2), int(t.height * k2)), Image.LANCZOS)
        bg = Image.new("RGB", (CW, CH), (255, 255, 255))
        bg.paste(t, ((CW - t.width) // 2, (CH - t.height) // 2))
        tiles.append(bg)

    cols = 3
    rr = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (CW * cols, rr * (CH + 76)), (255, 255, 255))
    dr = ImageDraw.Draw(sheet)
    for i, (r, t) in enumerate(zip(picked, tiles)):
        cx, cy = (i % cols) * CW, (i // cols) * (CH + 76)
        sheet.paste(t, (cx, cy))
        dr.text((cx + 8, cy + CH + 4), f"{i+1}. {r['rel']}", fill=(0, 0, 0), font=font(20, True))
        dr.text((cx + 8, cy + CH + 30),
                f"灰度 {r['root_gray']:.0f}　饱和 {r['root_sat']:.3f}　根宽 {r['root_width_px']:.2f}px",
                fill=(150, 0, 0), font=font(17))
        dr.text((cx + 8, cy + CH + 52),
                f"细根占比 {r['fine_pct']:.1f}%　阴影 {r['shadow_pct']:.2f}%　"
                f"贴边伪影 {r['edge_artifact']:.2f}%", fill=(90, 90, 90), font=font(16))
    o = OUT / "标准图集2_contact.png"
    sheet.save(o, optimize=True)
    print(f"\n  -> {o}\n  -> {OUT/'标准图集2.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
