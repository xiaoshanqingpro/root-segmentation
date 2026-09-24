"""处理 人工\9.9改\ 里的 7 张：按编号做出对应的机器图，放进 机器\9.9改mac\。

同时:
  - 随机抽 5 张典型图放到 测量\原图\
  - 新建 测试\调试\ 文件夹

机器图规格: 二值(纯黑白) + 未压缩 TIFF + 600dpi + 7019x4962，
            与之前送扫的 6 张完全一致，并附加"残留清理"。
参数显式设置（不依赖模块默认值，避免 45 号脚本踩过的坑）。
"""
from __future__ import annotations

import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

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
TEST = P.TEST
# 2026-09-23 用户把 采样数据\图片 移到了 扫描\原始图片\
SRC2 = P.RAW_D2
SRC99 = SRC2 / "多样性林" / "9.9"
PROJ = P.PROJ

# ---- 显式参数（与送扫的 6 张一致）----
R.FEAT_LONG = 3000
R.HALO_GROW = 0
R.SHRINK_PX = 1
R.SMOOTH_ALL = True
R.SMOOTH_OPEN_PX = 1
R.BRIDGE_CLOSE_PX = 2

# 残留清理参数
BIG = 2000
KEEP_DIST = 80
BAND_FRAC = 0.03
BAND_IN = 0.50
EDGE_DEPTH = 0.15
EDGE_MARGIN = 0.20       # 贴边块若整个外接框都在最外圈此比例内 -> 判扫描边        # 贴边块若垂直边界的尺寸 < 画幅此比例，判为沿边浅条（扫描伪影）          # 原 0.80 太高: 00000-001 有块 84025px 的贴边块带内占比 0.67 躲过了


def clean_residue(a: np.ndarray):
    """贴边成带的删、贴边浅条的删、远离主体的删、贴近主体的留。

    规则（2026-09-23 二次收紧）:
      1. 贴边 且 带内占比 >= BAND_IN                        -> 删（扫描仪盖板/黑边）
      2. 贴边 且 沿边方向"浅条"（垂直边界的尺寸 < EDGE_DEPTH 比例） -> 删
         —— 实测 7-002 顶部那条 1215x132 的横条就是这个形状: 细长、紧贴边、不往内延伸，
            是扫描边而不是根; 上一版因为它离主根只 80px 而被误留。
      3. 贴边 且 离主体 > KEEP_DIST                          -> 删
      4. 不贴边 且 面积<BIG 且 离主体>KEEP_DIST              -> 删（游离碎屑）
      5. 其余保留（靠近主体的碎块很可能是被管线打断的细根）
    """
    fg = a < 128
    H, W = fg.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), 8)
    band = np.zeros_like(fg)
    bh, bw = int(H * BAND_FRAC), int(W * BAND_FRAC)
    band[:bh, :] = band[-bh:, :] = band[:, :bw] = band[:, -bw:] = True
    drop = np.zeros_like(fg)
    se = {}

    big = [j for j in range(1, n) if st[j, cv2.CC_STAT_AREA] >= BIG]
    main = np.isin(lab, big)
    dist = cv2.distanceTransform((~main).astype(np.uint8), cv2.DIST_L2, 5)

    for j in range(1, n):
        x, y, w, h, area = st[j]
        m = lab == j
        t_top, t_bot, t_left, t_right = (y <= 0), (y + h >= H), (x <= 0), (x + w >= W)
        touch = t_top or t_bot or t_left or t_right
        d = float(dist[m].min())
        if touch:
            # 沿边浅条: 顶部/底部贴边时看高度，左右贴边时看宽度
            shallow = ((t_top or t_bot) and h < EDGE_DEPTH * H) or \
                      ((t_left or t_right) and w < EDGE_DEPTH * W)
            # 整个外接框都落在边角余量内 —— 抓"一端搭在根上"的扫描边（7-002 就躲在这）
            in_margin = (x + w <= EDGE_MARGIN * W) or (x >= (1 - EDGE_MARGIN) * W) or \
                        (y + h <= EDGE_MARGIN * H) or (y >= (1 - EDGE_MARGIN) * H)
            if float(band[m].mean()) >= BAND_IN:
                drop |= m
                se["edge"] = se.get("edge", 0) + int(area)
                continue
            if shallow or in_margin:
                drop |= m
                se["edge_strip"] = se.get("edge_strip", 0) + int(area)
                continue
            if d > KEEP_DIST:
                drop |= m
                se["edge_iso"] = se.get("edge_iso", 0) + int(area)
                continue
        elif area < BIG:
            if d > KEEP_DIST:
                drop |= m
                se["debris"] = se.get("debris", 0) + int(m.sum())
            else:
                se["kept"] = se.get("kept", 0) + int(m.sum())
    # 返回**图像**（根 = 0 黑、背景 = 255 白）。
    # 注意: 之前这里返回的是布尔掩膜（True=根），调用方直接当图像保存，
    # 结果 True→255(白) / False→0(黑)，输出成了反过来的「黑底白根」。
    keep = fg & ~drop
    return np.where(keep, 0, 255).astype(np.uint8), se


def orig_of(stem: str) -> Path | None:
    for e in (".tif", ".jpg", ".png"):
        p = SRC99 / (stem + e)
        if p.exists():
            return p
    return None


def main() -> int:
    # ---------- 1. 处理 9.9改 的 7 张 ----------
    man_dir = TEST / "人工" / "9.9改"
    mac_dir = TEST / "机器" / "9.9改mac"
    mac_dir.mkdir(parents=True, exist_ok=True)
    print("=== 1. 按编号生成机器图 -> 机器\\9.9改mac\\ ===")
    done = []
    for mf in sorted(man_dir.glob("*")):
        stem = mf.stem
        for suf in ("-man-9.22", "-man"):
            if stem.endswith(suf):
                stem = stem[: -len(suf)]
                break
        src = orig_of(stem)
        if src is None:
            # 兜底：在整个数据集2里找
            hits = list(SRC2.rglob(stem + ".*"))
            hits = [h for h in hits if "9.9改" not in h.parts]
            src = hits[0] if hits else None
        if src is None:
            print(f"  ! 找不到 {stem} 的原图，跳过")
            continue
        rgb = np.asarray(Image.open(src).convert("RGB"))
        root, _, info = R.extract_root(rgb, return_maps=False)
        binimg = np.where(root, 0, 255).astype(np.uint8)
        binimg, se = clean_residue(binimg)
        dst = mac_dir / f"{stem}-mac.tif"
        Image.fromarray(binimg).convert("RGB").save(
            dst, format="TIFF", compression="raw", dpi=(600, 600))
        done.append(stem)
        print(f"  {stem:<14} 源 {src.name:<16} 根 {info['root_pct']:6.3f}%  "
              f"删贴边 {se.get('edge',0):>6}px  删碎屑 {se.get('debris',0):>6}px  "
              f"留近块 {se.get('kept',0):>6}px  -> {dst.name}")

    # ---------- 2. 随机抽 5 张 -> 原图 ----------
    print("\n=== 2. 随机抽 5 张 -> 测试\\原图\\ ===")
    used = {"7-001", "7-002", "7-003", "66-003", "28-CL-003", "G218-001", "126-003",
            "134-003", "00000-001", "22-001", "44-001", "46-002"}
    pool = []
    for p in sorted(SRC2.rglob("*")):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".tif", ".tiff", ".png"):
            continue
        rel = p.relative_to(SRC2)
        if any(x in rel.parts for x in ("9.9改", "work", "ocr_raw")) or p.stem in used:
            continue
        pool.append(p)
    rng = random.Random(20260923)          # 固定种子，可复现
    pick = rng.sample(pool, 5)
    orig_dir = TEST / "原图"
    orig_dir.mkdir(parents=True, exist_ok=True)
    for p in pick:
        shutil.copy2(p, orig_dir / p.name)
        print(f"  {p.name:<20} <- {p.relative_to(SRC2)}")

    # ---------- 3. 新建 调试 文件夹 ----------
    dbg = TEST / "调试"
    dbg.mkdir(parents=True, exist_ok=True)
    print(f"\n=== 3. 新建文件夹 ===\n  {dbg}")

    print(f"\n  机器图 {len(done)} 张 -> {mac_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
