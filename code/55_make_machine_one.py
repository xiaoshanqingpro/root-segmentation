"""按样本号生成机器图（通用工具）。

把判根 + 残留清理 + 二值化打包成一个函数，以后要出图直接调。

用法:
    python 55_make_machine_one.py 281-003
    python 55_make_machine_one.py 281-003 182-001 224-001
    python 55_make_machine_one.py --all-orig      # 处理 测试\\原图\\ 里全部还没有机器图的
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
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

# ---- 冻结参数（与 9.9改mac / 送扫批次一致）----
R.FEAT_LONG = 3000
R.HALO_GROW = 0
R.SHRINK_PX = 1
R.SMOOTH_ALL = True
R.SMOOTH_OPEN_PX = 1
R.BRIDGE_CLOSE_PX = 2

# ---- 残留清理参数 ----
BIG = 2000
KEEP_DIST = 80
BAND_FRAC = 0.03
BAND_IN = 0.50
EDGE_DEPTH = 0.15
EDGE_MARGIN = 0.20
# 根是细长的，杂物是团块。用 sqrt(面积)/半宽 度量细长程度:
#   细长根 >= 3    实心团块 ~ 2
# 之前只按面积判"主体"，导致 G218-001/281-003 那种
# "面积够大的实心团块"被当成主体而豁免删除。
ELONG_ROOT = 3.0        # 达到此细长程度才算"根状"
BLOB_DIST = 30          # 团块离主体超过此距离 -> 判杂物
# 硬余量: 外接框整个落在最外圈此比例内 -> 一律判扫描边，**不要求接触边界**。
# 依据: 281-003 左下角那条竖条是扫描床/纸边，它不接触边界（左边有反光条挡着）、
#       又因为"细长"被当成根状主体豁免，前面所有规则都抓不到。
EDGE_MARGIN_HARD = 0.06


def clean_residue(img: np.ndarray):
    """贴边成带的删 / 贴边浅条的删 / 边角余量内的删 / 远离主体的删 / 贴近主体的留。

    入参与返回都是**图像**（根=0黑, 背景=255白），不是布尔掩膜。
    """
    fg = img < 128
    H, W = fg.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), 8)
    band = np.zeros_like(fg)
    bh, bw = int(H * BAND_FRAC), int(W * BAND_FRAC)
    band[:bh, :] = band[-bh:, :] = band[:, :bw] = band[:, -bw:] = True
    drop = np.zeros_like(fg)
    se: dict = {}

    # 每个块的细长程度
    elong: dict[int, float] = {}
    for j in range(1, n):
        sub = (lab == j)
        hw = float(cv2.distanceTransform(sub.astype(np.uint8), cv2.DIST_L2, 5).max())
        elong[j] = (float(st[j, cv2.CC_STAT_AREA]) ** 0.5) / max(0.5, hw)

    # 主体 = **最大的一块** 或 (面积>=BIG 且 细长像根)
    largest = max(range(1, n), key=lambda j: st[j, cv2.CC_STAT_AREA], default=0)
    main_ids = {j for j in range(1, n)
                if j == largest or (st[j, cv2.CC_STAT_AREA] >= BIG and elong[j] >= ELONG_ROOT)}
    main = np.isin(lab, list(main_ids))
    dist = cv2.distanceTransform((~main).astype(np.uint8), cv2.DIST_L2, 5)

    for j in range(1, n):
        x, y, w, h, area = st[j]
        m = lab == j
        t_top, t_bot, t_left, t_right = (y <= 0), (y + h >= H), (x <= 0), (x + w >= W)
        touch = t_top or t_bot or t_left or t_right
        d = float(dist[m].min())
        # 硬余量: 整个外接框在最外圈内 -> 一律判扫描边（放在最前面，不受主体豁免影响）
        if (x + w <= EDGE_MARGIN_HARD * W) or (x >= (1 - EDGE_MARGIN_HARD) * W) or \
           (y + h <= EDGE_MARGIN_HARD * H) or (y >= (1 - EDGE_MARGIN_HARD) * H):
            drop |= m
            se["edge_margin"] = se.get("edge_margin", 0) + int(area)
            continue
        if touch:
            shallow = ((t_top or t_bot) and h < EDGE_DEPTH * H) or \
                      ((t_left or t_right) and w < EDGE_DEPTH * W)
            in_margin = (x + w <= EDGE_MARGIN * W) or (x >= (1 - EDGE_MARGIN) * W) or \
                        (y + h <= EDGE_MARGIN * H) or (y >= (1 - EDGE_MARGIN) * H)
            if float(band[m].mean()) >= BAND_IN:
                drop |= m; se["edge"] = se.get("edge", 0) + int(area); continue
            if shallow or in_margin:
                drop |= m; se["edge_strip"] = se.get("edge_strip", 0) + int(area); continue
            if d > KEEP_DIST:
                drop |= m; se["edge_iso"] = se.get("edge_iso", 0) + int(area); continue
        else:
            if j in main_ids:
                continue
            if area < BIG and d > KEEP_DIST:
                drop |= m; se["debris"] = se.get("debris", 0) + int(area); continue
            # 大面积但形状是团块、且离开主体 -> 杂物（不是根）
            if elong[j] < ELONG_ROOT and d > BLOB_DIST:
                drop |= m; se["blob"] = se.get("blob", 0) + int(area); continue
            se["kept"] = se.get("kept", 0) + int(area)
    keep = fg & ~drop
    return np.where(keep, 0, 255).astype(np.uint8), se


def find_orig(stem: str) -> Path | None:
    """在 扫描\\原始图片 全域找同名原图（排除 9.9改）。"""
    for root in (P.RAW_D2, P.RAW_D1):
        if not root.exists():
            continue
        for e in (".jpg", ".jpeg", ".tif", ".tiff", ".png"):
            hits = [p for p in root.rglob(stem + e) if "9.9改" not in p.parts]
            if hits:
                return hits[0]
    return None


def make_machine(stem: str, out_dir: Path | None = None, quiet: bool = False) -> bool:
    src = find_orig(stem)
    if src is None:
        print(f"  ! 找不到 {stem} 的原图")
        return False
    out_dir = out_dir or P.TEST_MAC
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb = np.asarray(Image.open(src).convert("RGB"))
    root, _, info = R.extract_root(rgb, return_maps=False)
    img = np.where(root, 0, 255).astype(np.uint8)
    img, se = clean_residue(img)
    dst = out_dir / f"{stem}-mac.tif"
    Image.fromarray(img).convert("RGB").save(
        dst, format="TIFF", compression="raw", dpi=(600, 600))
    black = float((img == 0).mean() * 100)
    if not quiet:
        print(f"  {stem:<14} 源 {src.relative_to(P.SCAN)}")
        print(f"      判根 {info['root_pct']:6.3f}%  删贴边 {se.get('edge',0)+se.get('edge_strip',0)+se.get('edge_iso',0):>7}px"
              f"  删碎屑 {se.get('debris',0):>7}px  留近块 {se.get('kept',0):>7}px")
        print(f"      -> {dst.name}   根(黑)占比 {black:.2f}%   {img.shape[1]}x{img.shape[0]}")
    return True


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--all-orig" in sys.argv:
        stems = sorted(p.stem for p in P.TEST_ORIG.glob("*")
                       if p.suffix.lower() in (".jpg", ".jpeg", ".tif", ".tiff", ".png"))
        have = {p.stem.replace("-mac", "") for p in P.TEST_MAC.glob("*-mac.tif")}
        stems = [s for s in stems if s not in have]
        print(f"[批量] 测试\\原图 里还没有机器图的 {len(stems)} 张: {stems}")
    else:
        stems = args
    if not stems:
        print("用法: python 55_make_machine_one.py <样本号> [更多...]  或 --all-orig")
        return 1
    ok = sum(make_machine(s) for s in stems)
    print(f"\n  完成 {ok}/{len(stems)}   输出目录 {P.TEST_MAC}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
