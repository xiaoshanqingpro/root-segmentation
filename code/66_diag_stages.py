"""分阶段诊断：逐层复现 extract_root，看根在哪一层丢掉。

用法:
    python 66_diag_stages.py <样本名>
输出:
    outputs\\杉阔复核\\<样本>_分层\\NN_<阶段名>.png   每层 mask 预览（根=黑）
    控制台打印每层像素数 + 根所在区域的原始颜色
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import rootseg_common as R  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

Image.MAX_IMAGE_PIXELS = None
SRC = Path.home() / "Desktop" / "杉阔混交林"
OUTD = Path(r"D:\根系分割项目\outputs\杉阔复核")
EXTS = (".jpg", ".jpeg", ".tif", ".tiff", ".png")
FEAT_LONG = 3000
PW = 700

# 冻结参数（与 63_batch_shankuo.py 一致）
R.FEAT_LONG = FEAT_LONG
R.HALO_GROW = 0
R.SHRINK_PX = 1
R.SMOOTH_ALL = True
R.SMOOTH_OPEN_PX = 1
R.BRIDGE_CLOSE_PX = 2


def dump(name: str, m: np.ndarray, d: Path, note: str = "") -> None:
    small = Image.fromarray(np.where(m, 0, 255).astype(np.uint8)).convert("RGB")
    w = min(PW, small.width)
    small = small.resize((w, max(1, round(small.height * w / small.width))), Image.LANCZOS)
    small.save(d / f"{name}.png")
    print(f"    {name:<26} {int(m.sum()):>9} px   {m.mean()*100:7.4f}%   {note}")


def main(stem: str) -> int:
    p = next((q for q in SRC.rglob("*") if q.stem == stem and q.suffix.lower() in EXTS), None)
    if p is None:
        print(f"找不到 {stem}")
        return 1
    rgb = np.asarray(Image.open(p).convert("RGB"))
    H, W = rgb.shape[:2]
    sc = min(1.0, FEAT_LONG / max(H, W))
    small = cv2.resize(rgb, (int(W * sc), int(H * sc)), interpolation=cv2.INTER_AREA)
    hh, ww = small.shape[:2]
    d = OUTD / f"{stem}_分层"
    d.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {stem}  {p.relative_to(SRC)}  {W}x{H}  sc={sc:.4f}  特征尺度 {ww}x{hh} ===")

    bg = R.flood_background(small)
    fg = ~bg
    dump("01_bg洪水填充出的背景", bg, d, "白=背景")

    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1].astype(np.float32) / 255.0
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32)
    k = max(31, (int(min(hh, ww) * 0.05) // 2) * 2 + 1)
    illum = cv2.morphologyEx(gray.astype(np.uint8), cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(np.float32)
    depth = illum - gray
    corrected = np.clip(gray / np.maximum(illum, 1e-3) * float(illum.mean()), 0, 255)
    from skimage.filters import frangi as sk_frangi
    fr = sk_frangi((255.0 - corrected) / 255.0, sigmas=range(1, 6), black_ridges=False)
    fr = cv2.normalize(fr.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)

    deep = depth > R.DARK_DROP
    dark = gray < R.GRAY_TUB
    sat_ok = sat >= R.SAT_ROOT
    tub_ok = (fr > R.FRANGI_TUB) & dark
    core = fg & deep & (sat_ok | tub_ok)

    print(f"    (kernel k={k}, 前景 {fg.mean()*100:.2f}%)")
    dump("02_fg前景", fg, d)
    dump("03_deep够暗", fg & deep, d, f"depth>{R.DARK_DROP}")
    dump("04_饱和够褐", fg & deep & sat_ok, d, f"sat>={R.SAT_ROOT}")
    dump("05_管状响", fg & deep & tub_ok, d, f"fr>{R.FRANGI_TUB} & gray<{R.GRAY_TUB}")
    dump("06_core判根合计", core, d, "05+06 之前")

    # 形状救回
    pale = fg & (depth > R.PALE_DROP) & ~core
    np_, lp, sp, _ = cv2.connectedComponentsWithStats(pale.astype(np.uint8), 8)
    rescue = np.zeros_like(core)
    for j in range(1, np_):
        x, y, w_, h_, a = sp[j]
        if a < R.MIN_AREA_FEAT or max(w_, h_) < R.LEN_MIN:
            continue
        sub = (lp[y:y + h_, x:x + w_] == j)
        half_w = float(cv2.distanceTransform(sub.astype(np.uint8), cv2.DIST_L2, 5).max())
        if half_w > R.WIDTH_MAX:
            continue
        if (a ** 0.5) / max(0.5, half_w) < R.ELONG_MIN:
            continue
        if float(gray[lp == j].mean()) > R.PALE_GRAY_MAX:
            continue
        rescue |= (lp == j)
    dump("07_形状救回", rescue, d, f"{np_-1} 个淡前景块")
    core2 = core | rescue

    # 连通块过滤
    n, lab, stats, _ = cv2.connectedComponentsWithStats(core2.astype(np.uint8), 8)
    keep = np.zeros(n, bool)
    for j in range(1, n):
        if stats[j, cv2.CC_STAT_AREA] < R.MIN_AREA_FEAT:
            continue
        m = lab == j
        if rescue[m].mean() >= 0.5:
            keep[j] = True; continue
        if float(gray[m].mean()) > R.GRAY_COMP_MAX:
            continue
        if float(sat[m].mean()) < R.SAT_COMP_MIN:
            continue
        keep[j] = True
    core3 = keep[lab]
    dump("08_连通块过滤后", core3, d, f"{n-1} 块 -> 保留 {int(keep.sum())}")

    # 贴边伪影
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(core3.astype(np.uint8), 8)
    kill = np.zeros(n2, bool)
    for j in range(1, n2):
        x, y, w_, h_, a = st2[j]
        if not (x <= 1 or y <= 1 or x + w_ >= ww - 1 or y + h_ >= hh - 1):
            continue
        if (w_ >= R.EDGE_SPAN * ww) or (h_ >= R.EDGE_SPAN * hh) or (float(gray[lab2 == j].mean()) > R.EDGE_LIGHT):
            kill[j] = True
    core4 = core3 & ~kill[lab2]
    dump("09_贴边伪影剔除后", core4, d, f"剔除 {int(kill.sum())} 块")

    # 放大到全分辨率
    soft = cv2.resize(core4.astype(np.uint8), (W, H), interpolation=cv2.INTER_LINEAR)
    root = soft > 0.5
    fg_full = cv2.resize(fg.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
    root &= fg_full
    dump("10_放大回全分辨率", root, d)

    # 找"判根层漏掉的明显根"：在 fg 内但 core 外的、颜色偏褐的像素
    miss = fg_full & ~root
    print(f"\n    前景内未判根 {int(miss.sum())} px ({miss.mean()*100:.3f}%)")
    # 用参考图看这批漏掉像素的分布
    dump("11_前景内未判根", miss, d, "灰色区域=漏检候选")
    print(f"  分层图 -> {d}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "23-B-GG-002"))
