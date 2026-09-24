"""小闭环第 3 步：整图滑窗推理 + 与人工真值对比 + 可视化。

任务书阶段二要求"在这 10 张上推理，输出可视化对比图"。
这里用验证集 3 张整图（而非 patch）做推理，指标才有意义。

做法: 512 窗口 / 384 步长滑窗，概率图重叠区取平均，再阈值 0.5。
产出: outputs/训练/pred_<样本>.png   四格：原图 | 人工真值 | 模型预测 | 误差
      outputs/训练/infer_metrics.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
import paths as P  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
OUT = P.PROJ / "outputs" / "训练"
PATCH, STRIDE = 512, 384
NONWHITE = 250
THR = 0.5


def font(sz):
    for n in ("msyh.ttc", "simhei.ttf"):
        try:
            return ImageFont.truetype(n, sz)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


def find_orig(stem: str) -> Path | None:
    for d in [P.TEST_ORIG, P.RAW_D2_99, P.RAW_D2, P.RAW_D1]:
        if not d.exists():
            continue
        for p in d.rglob("*"):
            if p.suffix.lower() in (".tif", ".tiff", ".jpg", ".jpeg") and p.stem == stem:
                return p
    return None


def find_man(stem: str) -> Path | None:
    # 优先用自动清理过边框的版本（47_auto_clean_borders.py 产出），
    # 否则会把"已清边框的预测"拿去和"未清边框的真值"比，指标失真
    c = P.PROJ / "outputs" / "边框清理" / "cleaned" / f"{stem}-man-clean.tif"
    if c.exists():
        return c
    for d in [P.TEST_MAN_99GAI, P.TEST_MAN]:
        if not d.exists():
            continue
        for p in d.glob("*"):
            if "-man" in p.stem.lower() and p.stem.replace("-man", "").replace("-9.22", "") == stem:
                return p
    return None


@torch.no_grad()
def infer_full(model, img: np.ndarray, dev) -> np.ndarray:
    """滑窗整图推理，返回概率图（float32, HxW）"""
    H, W = img.shape[:2]
    prob = np.zeros((H, W), np.float32)
    cnt = np.zeros((H, W), np.float32)
    ys = list(range(0, max(1, H - PATCH) + 1, STRIDE))
    xs = list(range(0, max(1, W - PATCH) + 1, STRIDE))
    if ys[-1] != H - PATCH:
        ys.append(H - PATCH)
    if xs[-1] != W - PATCH:
        xs.append(W - PATCH)
    tiles, pos = [], []
    for y in ys:
        for x in xs:
            tiles.append(img[y:y + PATCH, x:x + PATCH])
            pos.append((y, x))
    for i in range(0, len(tiles), 8):
        batch = np.stack(tiles[i:i + 8]).astype(np.float32) / 255.0
        t = torch.from_numpy(batch).permute(0, 3, 1, 2).to(dev)
        with torch.amp.autocast("cuda", enabled=dev.type == "cuda"):
            out = model(t)
        p = torch.sigmoid(out).float().cpu().numpy()[:, 0]
        for k, (y, x) in enumerate(pos[i:i + 8]):
            prob[y:y + PATCH, x:x + PATCH] += p[k]
            cnt[y:y + PATCH, x:x + PATCH] += 1
    return prob / np.maximum(cnt, 1)


def main() -> int:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(OUT / "best_unet.pt", map_location="cpu", weights_only=False)
    import segmentation_models_pytorch as smp
    model = smp.Unet(encoder_name=ck["encoder"], encoder_weights=None,
                     in_channels=3, classes=1).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"[推理] 载入 {OUT/'best_unet.pt'}  (训练时 best val dice "
          f"{ck['val']['dice']:.4f} @ep{ck['epoch']})")

    import json
    summ = json.loads((OUT / "dataset_report.json").read_text(encoding="utf-8"))
    val_ids = summ["val_ids"]
    print(f"[推理] 验证集样本: {val_ids}")

    rows = []
    for stem in val_ids:
        o, m = find_orig(stem), find_man(stem)
        if not o or not m:
            print(f"  !! {stem} 缺文件，跳过")
            continue
        img = np.asarray(Image.open(o).convert("RGB"))
        man = np.asarray(Image.open(m).convert("RGB"))
        gt = np.any(man < NONWHITE, axis=2)
        print(f"  {stem}: 整图推理 {img.shape[1]}x{img.shape[0]} …", flush=True)
        prob = infer_full(model, img, dev)
        pred = prob > THR

        inter = float((gt & pred).sum())
        dice = 2 * inter / max(1e-6, gt.sum() + pred.sum())
        iou = inter / max(1e-6, (gt | pred).sum())
        prec = inter / max(1e-6, pred.sum())
        rec = inter / max(1e-6, gt.sum())
        rows.append({"样本": stem, "真值根%": round(100 * gt.mean(), 3),
                     "预测根%": round(100 * pred.mean(), 3),
                     "Dice": round(dice, 4), "IoU": round(iou, 4),
                     "精确": round(prec, 4), "召回": round(rec, 4)})
        print(f"    Dice {dice:.4f}  IoU {iou:.4f}  精确 {prec:.4f}  召回 {rec:.4f}  "
              f"(真值 {100*gt.mean():.3f}% / 预测 {100*pred.mean():.3f}%)")

        # 可视化
        ov = img.astype(np.float32)
        ov[gt & pred] = ov[gt & pred] * 0.35 + np.array([0, 175, 0]) * 0.65
        ov[~gt & pred] = ov[~gt & pred] * 0.2 + np.array([255, 40, 40]) * 0.8     # 误检
        ov[gt & ~pred] = ov[gt & ~pred] * 0.2 + np.array([40, 90, 255]) * 0.8     # 漏检
        panels = [img, np.stack([gt * 255] * 3, -1).astype(np.uint8),
                  np.stack([pred * 255] * 3, -1).astype(np.uint8), ov.astype(np.uint8)]
        PW = 620
        tiles = [cv2.resize(p, (PW, int(p.shape[0] * PW / p.shape[1])),
                            interpolation=cv2.INTER_AREA) for p in panels]
        row = np.hstack(tiles)
        c = Image.new("RGB", (row.shape[1], row.shape[0] + 48), (255, 255, 255))
        c.paste(Image.fromarray(row), (0, 48))
        dr = ImageDraw.Draw(c)
        dr.text((8, 6), f"U-Net 整图推理  {stem}   Dice {dice:.4f}  IoU {iou:.4f}  "
                        f"精确 {prec:.4f}  召回 {rec:.4f}", fill=(0, 0, 0), font=font(23))
        for i, t in enumerate(["原图", "人工真值", "模型预测", "误差(红=误检 蓝=漏检)"]):
            dr.text((8 + i * PW, 28), t, fill=(150, 0, 0), font=font(18))
        p = OUT / f"pred_{stem}.png"
        c.save(p, optimize=True)
        print(f"    -> {p}")

    if rows:
        import csv
        with (OUT / "infer_metrics.csv").open("w", newline="", encoding="utf-8-sig") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
        import statistics as st
        print(f"\n  验证集 3 张整图中位: Dice {st.median(r['Dice'] for r in rows):.4f}  "
              f"IoU {st.median(r['IoU'] for r in rows):.4f}  "
              f"精确 {st.median(r['精确'] for r in rows):.4f}  "
              f"召回 {st.median(r['召回'] for r in rows):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
