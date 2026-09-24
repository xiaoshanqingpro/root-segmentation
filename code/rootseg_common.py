"""共用的判根 / 白底化核心逻辑（数据集1 与 数据集2 同源同参）。

--- 2026-09-22 修订（针对用户反馈"有灰底残留、有小碎片、非根没全变白"）---

旧版的问题与根因:
  旧判据 core = 前景 ∩ 够暗 ∩ (管状响应 或 褐色饱和)
  其中"管状响应"用的是 Frangi，**没有附加"够暗"的约束**。
  而大块灰底的边界在某个尺度上同样是"脊状"，Frangi 照样亮 -> 灰块被整块判成根，保留原色 -> 残留。
  实测: 保留像素里 20~35% 是"偏亮+无彩"的灰底成分（某张高达 22.2%）。

新版做了四件事:
  1. 管状判据附加**绝对暗度**约束: fr 高 **且 灰度 < GRAY_TUB** 才算根
  2. 贴根半影外扩只允许"仍足够暗"的像素进来 (灰度 < GRAY_HALO)
  3. **连通块级过滤**: 面积过小、整体偏亮、整体无彩的块一律判背景
  4. 最终再按面积清一次碎块（阈值以**特征尺度像素**计，不再被 sc² 缩放抹平）

参数都集中在本文件顶部，两个数据集共用。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# ---- 与人工口径对齐 ----
FLOOD_TOL = 20          # 用户油漆桶容差
FEAT_LONG = 3000        # 特征尺度长边
                        # 原 1600。实测 6 张人工真值: 1400/2200/3000 三档下
                        # IoU 0.607 -> 0.663 -> 0.669，精确 0.714 -> 0.808 -> 0.825。
                        # 原因: 7019px 原图缩到 1400px 时缩放系数只有 0.2，
                        # 一根 3px 宽的细侧根只剩 0.7px，Frangi 在 sigmas 1~5 上看不见。
                        # 代价: 单张耗时约 14s（1400 时约 3.3s），8 进程并行下 180 张约 5 分钟。

# ---- 判根 ----
DARK_DROP = 12          # 比局部背景暗多少算"非背景"
SAT_ROOT = 0.10         # 褐色饱和度阈值
FRANGI_TUB = 0.20       # 管状响应阈值
GRAY_TUB = 220          # 管状判据的绝对暗度上限：比这亮就不算根
                        # 原值 170；实测 G218-001（根最淡那张）的细侧根灰度达 ~210，
                        # 被 170 挡掉。放宽到 220 —— 因为该判据仍要求"有管状响应"，
                        # 而噪声/灰块不是管状结构，不会因此混进来。
HALO_GROW = 6           # 贴根半影外扩上限(px, 特征尺度)
GRAY_HALO = 200         # 【新增】外扩只吸收比这暗的像素

# ---- 连通块级过滤 ----
MIN_AREA_FEAT = 25      # 【新增】特征尺度下最小面积，按此清碎块
GRAY_COMP_MAX = 185     # 【新增】块均灰度上限，超过判背景（灰块）
SAT_COMP_MIN = 0.045    # 【新增】块均饱和度下限，低于判背景（无彩灰块）

# ---- 形状救回：淡细根 vs 灰块 ----
# 难点: 淡细根与灰块**颜色几乎一样**（都浅、都低饱和），只能靠形状分。
# 淡细根细长；灰块成团。用 sqrt(面积)/半宽 作为"细长程度"的度量:
#   细长条  ~ 2*sqrt(长/宽)，细根可达 10 以上
#   团块    ~ sqrt(pi) ≈ 1.77
PALE_RESCUE = True
ELONG_MIN = 5.0         # 细长程度下限
LEN_MIN = 40            # 最长边下限（特征尺度 px）
WIDTH_MAX = 3.0         # 半宽上限（特征尺度 px）
PALE_GRAY_MAX = 232     # 救回对象的灰度上限（原 218；G218-001 的极淡细根到 220+ 被漏掉）
PALE_DROP = 6           # 【新增】救回路径专用的暗度门槛，比主判据的 12 更敏感
                        # 依据: 实测 G218-001（根最淡那张）的细侧根只比背景暗几个灰阶，
                        #       沿用主管道的 DARK_DROP=12 会先把它们滤掉，救回逻辑根本没机会看

# ---- 收边（全分辨率腐蚀）----
# 用用户精修真值（9.9改\7-001/2/3）实测发现:
#   原设置下我的根掩膜比用户的粗 1.6~1.9 倍，精确率仅 0.58
#   -> 直接导致根径高估 40~90%，根面积/体积跟着错
# 用户已确认"以精修真值为准"，故加入全分辨率腐蚀把掩膜收细。
SHRINK_PX = 0           # 全分辨率腐蚀半径（px）；0 = 不收缩

# ---- 抗锯齿 / 接断点（用户 2026-09-23 反馈）----
# 用户原话: "毛刺问题可以通过抗锯齿处理一下，还有接断点。"
# 毛刺来源: 掩膜在特征尺度(1600px)上算，放大回 7019px 时边界沿低分辨率像素格走，
#           形成锯齿状突起；WinRHIZO 把突起当根尖 -> 根尖数虚高(auto2 达 +88%)。
# 断点来源: 收边 + 阈值化会把细根打断 -> 每断一次多两个根尖。
# 处理: 先开运算去毛刺、再闭运算接断点，都在全分辨率上做，半径很小以免误伤细根。
SMOOTH_ALL = True
SMOOTH_OPEN_PX = 1      # 开运算半径: 去掉 <=2px 的锯齿突起
BRIDGE_CLOSE_PX = 2     # 闭运算半径: 接上 <=4px 的断裂

# ---- 贴边伪影 ----
EDGE_KILL = True
EDGE_SPAN = 0.25        # 贴边且外接框跨度超过画幅此比例 -> 伪影
EDGE_LIGHT = 195        # 贴边且整体偏亮 -> 伪影（纸边/盖板边）

JPEG_Q = 95


def flood_background(small: np.ndarray, tol: int = FLOOD_TOL) -> np.ndarray:
    """背景 = 边界种子 + 内部亮点种子的洪水填充。

    补内部亮点种子是必须的: 第二数据集四边有扫描仪黑边，
    只从边界播种会把黑边填成背景，而白纸反而不与边界连通 -> 整张图全成前景。
    """
    hh, ww = small.shape[:2]
    mask = np.zeros((hh + 2, ww + 2), np.uint8)
    flags = 4 | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | (255 << 8)
    img = np.ascontiguousarray(small)
    step = max(1, min(hh, ww) // 64)
    gray0 = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    bright = gray0 > max(180, int(np.percentile(gray0, 80)))
    seeds = [(x, 0) for x in range(0, ww, step)] + [(x, hh - 1) for x in range(0, ww, step)] \
        + [(0, y) for y in range(0, hh, step)] + [(ww - 1, y) for y in range(0, hh, step)]
    for y in range(step, hh - step, max(1, hh // 24)):
        for x in range(step, ww - step, max(1, ww // 24)):
            if bright[y, x]:
                seeds.append((x, y))
    for (sx, sy) in seeds:
        if not mask[sy + 1, sx + 1]:
            cv2.floodFill(img, mask, (sx, sy), 255, (tol,) * 3, (tol,) * 3, flags)
    return mask[1:-1, 1:-1] > 0


def diagnose(gray, sat, fr, fg, core, sc, tag=""):
    """统计判根质量：灰底残留比例等（用于回归检查）。"""
    if core.sum() == 0:
        return {}
    g = gray[core]
    s = sat[core]
    light = float((g > 180).mean() * 100)
    nocol = float((s < 0.06).mean() * 100)
    both = float(((g > 180) & (s < 0.06)).mean() * 100)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(core.astype(np.uint8), 8)
    areas = np.array([stats[j, cv2.CC_STAT_AREA] for j in range(1, n)])
    small = int((areas < 500).sum()) if len(areas) else 0
    return {"tag": tag, "root_pct": round(float(core.mean() * 100), 3),
            "bright_pct": round(light, 1), "colorless_pct": round(nocol, 1),
            "graybg_pct": round(both, 1), "components": n - 1, "tiny_components": small}


def extract_root(rgb: np.ndarray, feat_long: int = FEAT_LONG, return_maps: bool = False,
                 apply_post: bool = True):
    """从 RGB 图提取根掩膜（全分辨率）。返回 (root_full, fg_full, info)

    apply_post=False 时跳过"收边"与"抗锯齿/接断点"，返回这两步之前的原始掩膜。
    参数扫描时用它只算一次特征，再对原始掩膜反复施加不同的后处理，省大量时间。
    """
    H, W = rgb.shape[:2]
    sc = min(1.0, feat_long / max(H, W))
    small = cv2.resize(rgb, (int(W * sc), int(H * sc)), interpolation=cv2.INTER_AREA) if sc < 1 else rgb
    hh, ww = small.shape[:2]

    bg = flood_background(small)
    fg = ~bg

    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1].astype(np.float32) / 255.0
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32)
    k = max(31, (int(min(hh, ww) * 0.05) // 2) * 2 + 1)
    illum = cv2.morphologyEx(gray.astype(np.uint8), cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(np.float32)
    depth = illum - gray
    corrected = np.clip(gray / np.maximum(illum, 1e-3) * float(illum.mean()), 0, 255)
    try:
        from skimage.filters import frangi as sk_frangi
        fr = sk_frangi((255.0 - corrected) / 255.0, sigmas=range(1, 6), black_ridges=False)
        fr = cv2.normalize(fr.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
    except Exception:  # noqa: BLE001
        fr = np.zeros_like(gray)

    deep = depth > DARK_DROP
    dark = gray < GRAY_TUB                                  # 【新增 1】管状判据要够暗
    core = fg & deep & ((sat >= SAT_ROOT) | ((fr > FRANGI_TUB) & dark))

    # 【新增 1.5】形状救回：颜色不够但**细长**的浅色细根，仍判根
    # 先看剩下的前景里，哪些是"细长条"而不是"团块"
    rescued = np.zeros_like(core)      # 记录被救回的像素，供后面过滤时豁免
    if PALE_RESCUE:
        pale = fg & (depth > PALE_DROP) & ~core
        np_, lp, sp, _ = cv2.connectedComponentsWithStats(pale.astype(np.uint8), 8)
        rescue = np.zeros_like(core)
        for j in range(1, np_):
            x, y, w_, h_, a = sp[j]
            if a < MIN_AREA_FEAT or max(w_, h_) < LEN_MIN:
                continue
            sub = (lp[y:y + h_, x:x + w_] == j)
            half_w = float(cv2.distanceTransform(sub.astype(np.uint8), cv2.DIST_L2, 5).max())
            if half_w > WIDTH_MAX:
                continue
            elong = (a ** 0.5) / max(0.5, half_w)
            if elong < ELONG_MIN:
                continue
            if float(gray[lp == j].mean()) > PALE_GRAY_MAX:
                continue
            rescue |= (lp == j)
        core = core | rescue
        rescued = rescue.copy()

    # 【新增 2】贴根半影外扩：只吸收仍足够暗的像素
    if HALO_GROW > 0:
        g = max(1, int(round(HALO_GROW * sc)))
        grown = cv2.dilate(core.astype(np.uint8),
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * g + 1))) > 0
        core = core | (grown & fg & deep & (gray < GRAY_HALO))

    # 【新增 3】连通块级过滤：面积小 / 整体偏亮 / 整体无彩 -> 背景
    #   注意: 被"形状救回"的块必须**豁免**颜色判据。
    #   否则会自相矛盾: 救回逻辑刚把浅色细根捞回来，
    #   紧接着"块均灰度>185 判背景"又按颜色把它删掉 —— 实测 G218-001 就是这么丢的。
    n, lab, stats, _ = cv2.connectedComponentsWithStats(core.astype(np.uint8), 8)
    keep = np.zeros(n, bool)
    for j in range(1, n):
        if stats[j, cv2.CC_STAT_AREA] < MIN_AREA_FEAT:
            continue
        m = lab == j
        if rescued[m].mean() >= 0.5:      # 主体是被救回的细长结构 -> 直接保留
            keep[j] = True
            continue
        if float(gray[m].mean()) > GRAY_COMP_MAX:
            continue
        if float(sat[m].mean()) < SAT_COMP_MIN:
            continue
        keep[j] = True
    core = keep[lab]

    # 【新增 4】贴边伪影：贴边且跨度大 / 整体偏亮
    if EDGE_KILL:
        n2, lab2, st2, _ = cv2.connectedComponentsWithStats(core.astype(np.uint8), 8)
        kill = np.zeros(n2, bool)
        for j in range(1, n2):
            x, y, w_, h_, a = st2[j]
            touches = (x <= 1 or y <= 1 or x + w_ >= ww - 1 or y + h_ >= hh - 1)
            if not touches:
                continue
            if (w_ >= EDGE_SPAN * ww) or (h_ >= EDGE_SPAN * hh) or (float(gray[lab2 == j].mean()) > EDGE_LIGHT):
                kill[j] = True
        core = core & ~kill[lab2]

    if sc < 1.0:
        # 掩膜在特征尺度上算，放大回全分辨率时用**平滑插值再阈值**，
        # 避免最近邻放大产生明显锯齿（测量根径时锯齿会直接影响边缘）
        # 注意: core 是 0/1 的 uint8，插值后取值范围是 [0,1]，阈值必须用 0.5 而不是 127
        soft = cv2.resize(core.astype(np.uint8), (W, H), interpolation=cv2.INTER_LINEAR)
        root_full = soft > 0.5
        fg_full = cv2.resize(fg.astype(np.uint8), (W, H),
                             interpolation=cv2.INTER_NEAREST).astype(bool)
        root_full &= fg_full
    else:
        root_full, fg_full = core, fg

    # 收边：在全分辨率上腐蚀，把比真值偏粗的部分收回去
    if apply_post and SHRINK_PX > 0:
        ker_s = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * SHRINK_PX + 1,) * 2)
        root_full = cv2.erode(root_full.astype(np.uint8), ker_s) > 0

    # 抗锯齿 + 接断点（全分辨率）
    #   先闭运算接断点（把收边造成的细断裂接上），再开运算去毛刺（去掉锯齿状突起）。
    #   顺序很重要: 先闭后开 = 形态学"平滑"，既补断又去毛刺；反过来会把断口拉大。
    if apply_post and SMOOTH_ALL:
        u8 = root_full.astype(np.uint8)
        if BRIDGE_CLOSE_PX > 0:
            kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * BRIDGE_CLOSE_PX + 1,) * 2)
            u8 = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, kc)
        if SMOOTH_OPEN_PX > 0:
            ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * SMOOTH_OPEN_PX + 1,) * 2)
            u8 = cv2.morphologyEx(u8, cv2.MORPH_OPEN, ko)
        root_full = u8 > 0

    info = {"sc": sc, "fg_pct": round(float(fg_full.mean() * 100), 3),
            "root_pct": round(float(root_full.mean() * 100), 3)}
    if return_maps:
        info["diag_feat"] = diagnose(gray, sat, fr, fg, core, sc)
    return root_full, fg_full, info


def whiten(rgb: np.ndarray, root_full: np.ndarray) -> np.ndarray:
    """非根 -> 纯白；根保留原色。

    就地改写传入的 rgb 以省一份全分辨率拷贝（7019x4962 的 RGB 就是 104 MB，
    多进程并行时这一份内存很关键）。
    """
    out = rgb if rgb.flags.writeable else rgb.copy()
    out[~root_full] = 255
    return out


# ---------------------------------------------------------------------------
# 单文件处理（供多进程并行调用）
# ---------------------------------------------------------------------------
def process_file(args) -> dict:
    """args = (src_path, out_path, preview_path, make_preview)

    注意: 这是给 ProcessPoolExecutor 用的顶层函数，参数必须可 pickle，
    所以传字符串路径而不是 Path。
    """
    import time
    from pathlib import Path as _P

    src, dst, prev, make_prev = args
    # 每个进程内部关掉 OpenCV 自带的多线程：32 核 x N 线程会互相抢，
    # 并行度交给进程池控制更稳。
    cv2.setNumThreads(1)
    t0 = time.time()
    src_p, dst_p = _P(src), _P(dst)
    rgb = np.asarray(Image.open(src_p).convert("RGB"))
    root, fg, info = extract_root(rgb, return_maps=True)
    out = whiten(rgb, root)
    dst_p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out).save(dst_p, "JPEG", quality=JPEG_Q, subsampling=0, dpi=(600, 600))
    if make_prev and prev:
        _write_preview(_P(prev), out, src_p.name, info)
    d = info["diag_feat"]
    return {"file": src_p.name, "out_path": str(dst_p), "fg_pct": info["fg_pct"],
            "root_pct": info["root_pct"], "graybg_pct": d["graybg_pct"],
            "tiny_components": d["tiny_components"], "seconds": round(time.time() - t0, 1)}


def _write_preview(path, out: np.ndarray, name: str, info: dict) -> None:
    from PIL import ImageDraw, ImageFont
    pv = Image.fromarray(out)
    PW = 900
    pv = pv.resize((PW, int(pv.height * PW / pv.width)), Image.LANCZOS)
    canvas = Image.new("RGB", (PW, pv.height + 56), (255, 255, 255))
    canvas.paste(pv, (0, 56))
    dr = ImageDraw.Draw(canvas)
    f1 = f2 = None
    for n in ("msyh.ttc", "simhei.ttf"):
        try:
            f1 = ImageFont.truetype(n, 21); f2 = ImageFont.truetype(n, 18); break
        except Exception:  # noqa: BLE001
            continue
    if f1 is None:
        f1 = f2 = ImageFont.load_default()
    d = info.get("diag_feat", {})
    dr.text((8, 6), f'白底根图  {name}   {info["root_pct"]:.3f}% 判为根', fill=(0, 0, 0), font=f1)
    dr.text((8, 30), f'前景 {info["fg_pct"]:.2f}%   灰底残留 {d.get("graybg_pct", 0):.1f}%   '
                     f'碎块 {d.get("tiny_components", 0)} 个', fill=(150, 0, 0), font=f2)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, optimize=True)


def run_parallel(tasks: list, workers: int = 8, label: str = "") -> list:
    """多进程跑一批任务，带进度输出。"""
    import time
    from concurrent.futures import ProcessPoolExecutor, as_completed

    rows = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(process_file, t): t for t in tasks}
        for k, fu in enumerate(as_completed(futs), 1):
            try:
                rows.append(fu.result())
            except Exception as exc:  # noqa: BLE001
                rows.append({"file": str(futs[fu][0]), "error": f"{type(exc).__name__}: {exc}"})
            if k % 5 == 0 or k == len(tasks):
                el = time.time() - t0
                print(f"[{label} {k}/{len(tasks)}] 已用 {el:.0f}s  "
                      f"均 {el/k:.1f}s/张  预计剩余 {el/k*(len(tasks)-k):.0f}s", flush=True)
    return rows
