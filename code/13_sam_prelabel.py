"""阶段二 · SAM v1 预标注（路线乙）：滑窗跑，输出三分类候选 mask 供人工修正。

为什么是 SAM v1 而不是 SAM2:
  SAM2 在 PyPI 只有源码包，本机无 MSVC C++ 工具链 -> CUDA 算子编译不了，装不上。
  segment-anything (SAM v1) 是纯 PyTorch，免编译，是本机唯一可行的 SAM 路线。

为什么必须滑窗（而不是整图丢给 SAM）:
  实测整图跑直接 CUDA OOM: SAM 的图像编码器固定 1024，但 postprocess 要把 256x256 的
  低分掩膜上采样回**原图尺寸**(4900x6900)。默认 points_per_batch=64 时，
  单次 interpolate 就要 24.66 GiB >> 本机 7.96 GiB 显存。
  两条出路: (a) 缩小 points_per_batch —— 能跑但很慢, 掩膜仍是全分辨率;
           (b) **按 1024 窗口滑窗** —— 每个窗口的 original_size 就是 1024,
               既不会 OOM, 又**不牺牲边界精度**(根部细到 4-5px, 缩放会直接糊掉)。
  这里选 (b)。整图缩放会把 0.2mm 的细根从 4.7px 压到 2px, 对根径测量是致命的。

候选不是标签: 人工修正后的才是训练标签, 存到 labels/mask_v1/。

产出:
  labels/mask_v0_candidate/<原文件名>_mask.png        8bit (0=背景 1=根 2=阴影)
  outputs/阶段二/candidates_preview/<原文件名>_overlay.png
  outputs/阶段二/prelabel_report.csv
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# --- 统一路径（见 paths.py；数据搬家只改那里）---
sys.path.insert(0, str(Path(__file__).parent))
import paths as P  # noqa: E402

PROJ = P.PROJ
CFG = PROJ / "config" / "phase2_images.json"
CKPT = PROJ / "models" / "sam_vit_b_01ec64.pth"
LABEL_DIR = PROJ / "labels" / "mask_v0_candidate"
PREVIEW_DIR = PROJ / "outputs" / "阶段二" / "candidates_preview"
REPORT = PROJ / "outputs" / "阶段二" / "prelabel_report.csv"

TILE = 1024
STRIDE = 896              # 128px 重叠，压掉拼接缝
PTS_SIDE = 16
PTS_BATCH = 16
FEAT_LONG = 1200
DARK_DROP = 14
SAT_ROOT = 0.12
FRANGI_TUB = 0.22
SKIP_IF_DARK_LT = 0.001   # 整块几乎全白就跳过，省时间
SHADOW_FROM_CLASSICAL = True   # 阴影用经典照度模型；交给 SAM 会产生窗口形状的方块伪影
CLASSICAL_ROOT_UNION = True    # 根 = SAM ∪ 经典根芯；SAM 会漏细根，候选宁可多不可少
RAW_DIR = PROJ / "outputs" / "阶段二" / "_sam_raw"   # 缓存 SAM 中间结果，改组装逻辑时不必重跑


def font(sz):
    for n in ("msyh.ttc", "simhei.ttf"):
        try:
            return ImageFont.truetype(n, sz)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


def feature_maps(rgb: np.ndarray):
    """降到长边 FEAT_LONG 算特征：掩膜级统计只要均值，不需要全分辨率。"""
    h, w = rgb.shape[:2]
    sc = FEAT_LONG / max(h, w)
    if sc < 1.0:
        small = cv2.resize(rgb, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
    else:
        sc, small = 1.0, rgb
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1].astype(np.float32) / 255.0
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32)
    k = max(31, (int(min(gray.shape) * 0.05) // 2) * 2 + 1)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    illum = cv2.morphologyEx(gray.astype(np.uint8), cv2.MORPH_CLOSE, ker).astype(np.float32)
    depth = illum - gray
    corrected = np.clip(gray / np.maximum(illum, 1e-3) * float(illum.mean()), 0, 255)
    inv = (255.0 - corrected) / 255.0
    try:
        from skimage.filters import frangi as sk_frangi
        fr = sk_frangi(inv, sigmas=range(1, 6), black_ridges=False)
        fr = cv2.normalize(fr.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
    except Exception:  # noqa: BLE001
        fr = np.zeros_like(gray)
    return depth, sat, fr, sc


def classify(depth, sat, fr, mask_full, scale):
    """按标准给一个全分辨率掩膜定类 -> (类, 统计)"""
    if scale < 1.0:
        m = cv2.resize(mask_full.astype(np.uint8), (depth.shape[1], depth.shape[0]),
                       interpolation=cv2.INTER_NEAREST).astype(bool)
    else:
        m = mask_full
    if m.sum() < 24:
        return 0, None
    d = float(depth[m].mean())
    s = float(sat[m].mean())
    f = float((fr[m] > FRANGI_TUB).mean())
    st = {"depth": round(d, 2), "sat": round(s, 4), "tub": round(f, 4)}
    if d > DARK_DROP and (f > 0.06 or s >= SAT_ROOT):
        return 1, st          # 根
    if d > DARK_DROP and s < SAT_ROOT and f <= 0.06:
        return 2, st          # 阴影
    return 0, st


def main() -> int:
    if not CKPT.exists():
        raise SystemExit(f"缺少 SAM 权重 {CKPT}")

    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[prelabel] device={dev}  ckpt={CKPT.name} ({CKPT.stat().st_size/1024**2:.1f} MB)", flush=True)
    print(f"[prelabel] 滑窗 {TILE}px / 步长 {STRIDE}px / {PTS_SIDE}x{PTS_SIDE} 提示点", flush=True)

    sam = sam_model_registry["vit_b"](checkpoint=str(CKPT)).to(device=dev)
    gen = SamAutomaticMaskGenerator(
        sam,
        points_per_side=PTS_SIDE,
        points_per_batch=PTS_BATCH,
        pred_iou_thresh=0.85,
        stability_score_thresh=0.88,
        crop_n_layers=0,
        min_mask_region_area=0,
    )

    cfg = json.loads(CFG.read_text(encoding="utf-8"))
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for idx, it in enumerate(cfg["items"], 1):
        p = PROJ / it["processed_path"]
        t0 = time.time()
        rgb = np.asarray(Image.open(p).convert("RGB"))
        H, W = rgb.shape[:2]
        depth, sat, fr, scale = feature_maps(rgb)
        dark_small = (depth > DARK_DROP)

        label = np.zeros((H, W), np.uint8)
        sam_root = np.zeros((H, W), np.uint8)      # 只记 SAM 判为根的像素
        raw_path = RAW_DIR / f"{p.stem}_samroot.png"
        n_tiles = n_skip = n_masks = 0
        cls_masks = {1: 0}

        if raw_path.exists():
            sam_root = np.asarray(Image.open(raw_path))
            print(f"[{idx}/{len(cfg['items'])}] {p.name}: 复用已缓存的 SAM 结果", flush=True)
            n_tiles = n_masks = -1
        else:
            ys = list(range(0, max(1, H - TILE + STRIDE), STRIDE)) or [0]
            xs = list(range(0, max(1, W - TILE + STRIDE), STRIDE)) or [0]
            for y0 in ys:
                for x0 in xs:
                    y1, x1 = min(y0 + TILE, H), min(x0 + TILE, W)
                    y0c, x0c = max(0, y1 - TILE), max(0, x1 - TILE)
                    n_tiles += 1
                    fy0, fy1 = int(y0c * scale), int(np.ceil(y1 * scale))
                    fx0, fx1 = int(x0c * scale), int(np.ceil(x1 * scale))
                    sub = dark_small[fy0:fy1, fx0:fx1]
                    if sub.size == 0 or sub.mean() < SKIP_IF_DARK_LT:
                        n_skip += 1
                        continue

                    tile = np.ascontiguousarray(rgb[y0c:y1, x0c:x1])
                    masks = gen.generate(tile)
                    n_masks += len(masks)
                    masks.sort(key=lambda m: -m["area"])
                    for m in masks:
                        seg = m["segmentation"]
                        full = np.zeros((H, W), bool)
                        full[y0c:y0c + seg.shape[0], x0c:x0c + seg.shape[1]] = seg
                        c, st = classify(depth, sat, fr, full, scale)
                        if c != 1:
                            continue
                        cls_masks[1] += 1
                        sam_root[y0c:y0c + seg.shape[0], x0c:x0c + seg.shape[1]][seg] = 255
            Image.fromarray(sam_root).save(raw_path, optimize=True)

        # ---- 组装三类标签 ----
        # 根 = SAM 精确边界  ∪  经典方法的根芯
        #   理由: SAM 边界准但会漏细根（实测覆盖率只有经典的 40-60%）；
        #         经典方法召回高（90%+）但掩膜偏"根芯"、比真实根细，且带阴影误检。
        #         候选宁可多不可少 —— 人工"擦掉"比"补画"省力得多。
        label[sam_root > 0] = 1
        if CLASSICAL_ROOT_UNION:
            croot = (depth > DARK_DROP) & ((fr > FRANGI_TUB) | (sat >= SAT_ROOT))
            croot = cv2.morphologyEx(croot.astype(np.uint8), cv2.MORPH_OPEN,
                                     np.ones((3, 3), np.uint8)) > 0
            if scale != 1.0:
                croot = cv2.resize(croot.astype(np.uint8), (W, H),
                                   interpolation=cv2.INTER_NEAREST).astype(bool)
            label[croot] = 1

        # 阴影: 回归经典照度模型（SAM 判背景块会把窗口形状带进来，形成方块伪影）
        if SHADOW_FROM_CLASSICAL:
            corr = (depth > DARK_DROP) & (sat < SAT_ROOT) & (fr <= FRANGI_TUB)
            corr_u8 = cv2.morphologyEx(corr.astype(np.uint8), cv2.MORPH_OPEN,
                                       np.ones((5, 5), np.uint8))
            corr_u8 = cv2.morphologyEx(corr_u8, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
            sh_small = corr_u8 > 0
            if scale != 1.0:
                sh_full = cv2.resize(sh_small.astype(np.uint8), (W, H),
                                     interpolation=cv2.INTER_NEAREST).astype(bool)
            else:
                sh_full = sh_small
            sh_full &= (label == 0)                # 根优先，阴影不覆盖根
            label[sh_full] = 2

        out_mask = LABEL_DIR / f"{p.stem}_mask.png"
        Image.fromarray(label).save(out_mask, optimize=True)

        ov = rgb.astype(np.float32)
        ov[label == 1] = ov[label == 1] * 0.45 + np.array([0, 200, 0]) * 0.55
        ov[label == 2] = ov[label == 2] * 0.55 + np.array([255, 40, 40]) * 0.45
        prev = Image.fromarray(ov.astype(np.uint8))
        PW = 900
        prev = prev.resize((PW, int(prev.height * PW / prev.width)), Image.LANCZOS)
        canvas = Image.new("RGB", (PW, prev.height + 44), (255, 255, 255))
        canvas.paste(prev, (0, 44))
        dr = ImageDraw.Draw(canvas)
        dr.text((8, 6), f'SAM v1 候选  {p.name}', fill=(0, 0, 0), font=font(21))
        dr.text((8, 26), f'绿=根 {100*(label==1).mean():.2f}%   红=阴影 {100*(label==2).mean():.2f}%   '
                         f'窗口 {n_tiles}（跳过 {n_skip}）  {time.time()-t0:.0f}s',
                fill=(150, 0, 0), font=font(18))
        canvas.save(PREVIEW_DIR / f"{p.stem}_overlay.png", optimize=True)

        rows.append({
            "file": p.name, "rel_path": it["rel_path"],
            "tiles": n_tiles, "tiles_skipped": n_skip, "sam_masks": n_masks,
            "masks_root": cls_masks[1],
            "root_pct": round(float((label == 1).mean() * 100), 3),
            "shadow_pct": round(float((label == 2).mean() * 100), 3),
            "seconds": round(time.time() - t0, 1),
            "label_path": str(out_mask.relative_to(PROJ)),
        })
        print(f"[{idx}/{len(cfg['items'])}] {p.name:<18} 窗口 {n_tiles}(跳过{n_skip})  "
              f"SAM掩膜 {n_masks} -> 根类 {cls_masks[1]}  "
              f"像素 根 {rows[-1]['root_pct']}% 阴影 {rows[-1]['shadow_pct']}%  "
              f"{rows[-1]['seconds']}s", flush=True)

    with REPORT.open("w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"\n  候选标签 -> {LABEL_DIR}")
    print(f"  预览图   -> {PREVIEW_DIR}")
    print(f"  报告     -> {REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
