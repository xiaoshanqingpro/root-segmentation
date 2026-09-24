"""小闭环训练：U-Net 两类分割（根 / 非根）。

按任务书阶段二要求:
  - 512x512 patch
  - Dice + CE 损失
  - 数据增强：水平/垂直翻转、90°旋转、±20% 亮度抖动
  - 按整图划分 train/val（同一张图的 patch 不跨集合）—— 已在 42_build_dataset.py 落实
  - 固定随机种子

本脚本只做"流程跑得通、输出看起来像那么回事"，不为刷指标调参。

产出:
  outputs/训练/best_unet.pt          最佳权重
  outputs/训练/train_log.csv         逐 epoch 指标
  outputs/训练/training_curve.png    曲线
  outputs/训练/pred_<样本>.png        验证集预测可视化
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
import paths as P  # noqa: E402

OUT = P.PROJ / "outputs" / "训练"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42


def font(sz):
    for n in ("msyh.ttc", "simhei.ttf"):
        try:
            return ImageFont.truetype(n, sz)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


def set_seed(s: int) -> None:
    import random
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_augment(training: bool):
    import albumentations as A
    if training:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),                      # 90° 旋转
            A.RandomBrightnessContrast(brightness_limit=0.20,   # ±20% 亮度抖动
                                       contrast_limit=0.20, p=0.5),
        ])
    return None


class PatchDS(torch.utils.data.Dataset):
    def __init__(self, X, Y, train: bool):
        self.X, self.Y = X, Y
        self.aug = get_augment(train)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = self.X[i]
        y = self.Y[i]
        if self.aug is not None:
            r = self.aug(image=x, mask=y)
            x, y = r["image"], r["mask"]
        x = torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1).float() / 255.0
        y = torch.from_numpy(np.ascontiguousarray(y)).float()
        return x, y


def dice_loss(logits, target, eps=1e-6):
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum(dim=(1, 2)) + eps
    den = p.sum(dim=(1, 2)) + target.sum(dim=(1, 2)) + eps
    return 1 - (num / den).mean()


class ComboLoss(nn.Module):
    """Dice + BCE（二值版的 Dice + CE）"""
    def __init__(self, w_dice=0.5, w_ce=0.5):
        super().__init__()
        self.w_dice, self.w_ce = w_dice, w_ce
        self.ce = nn.BCEWithLogitsLoss()

    def forward(self, logits, target):
        return self.w_dice * dice_loss(logits, target) + \
               self.w_ce * self.ce(logits, target)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    inter = union = tp = fp = fn = 0
    t0 = time.time()
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        if logits.shape[-2:] != y.shape[-2:]:
            logits = nn.functional.interpolate(logits, size=y.shape[-2:], mode="bilinear",
                                               align_corners=False)
        pred = (torch.sigmoid(logits) > 0.5).float().cpu()
        y = y.unsqueeze(1)
        inter += float((pred * y).sum())
        union += float(((pred + y) > 0).sum())
        tp += float((pred * y).sum())
        fp += float((pred * (1 - y)).sum())
        fn += float(((1 - pred) * y).sum())
    dice = 2 * tp / max(1e-6, 2 * tp + fp + fn)
    iou = inter / max(1e-6, union)
    prec = tp / max(1e-6, tp + fp)
    rec = tp / max(1e-6, tp + fn)
    return {"dice": dice, "iou": iou, "precision": prec, "recall": rec,
            "sec": round(time.time() - t0, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--encoder", default="resnet34")
    # 注意：Windows 上 num_workers>0 会死锁（实测卡在 ep30 两小时不动，CPU 零增长）。
    # 本数据集只有 ~76MB、全在内存里，worker 进程毫无必要，固定 0。
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="只跑 2 个 epoch 验证流程")
    a = ap.parse_args()
    if a.smoke:
        a.epochs = 2

    set_seed(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[训练] device={dev}  encoder={a.encoder}  epochs={a.epochs}  batch={a.batch}")

    dtr = np.load(OUT / "patches_train.npz", allow_pickle=True)
    dva = np.load(OUT / "patches_val.npz", allow_pickle=True)
    Xtr, Ytr = dtr["X"], dtr["Y"]
    Xva, Yva = dva["X"], dva["Y"]
    print(f"[训练] train {Xtr.shape}   val {Xva.shape}")
    print(f"[训练] train 根占比 {Ytr.mean()*100:.3f}%   val 根占比 {Yva.mean()*100:.3f}%")

    tr_ld = torch.utils.data.DataLoader(PatchDS(Xtr, Ytr, True), batch_size=a.batch,
                                        shuffle=True, num_workers=a.workers, drop_last=False,
                                        pin_memory=True)
    va_ld = torch.utils.data.DataLoader(PatchDS(Xva, Yva, False), batch_size=a.batch,
                                        shuffle=False, num_workers=a.workers, pin_memory=True)

    import segmentation_models_pytorch as smp
    model = smp.Unet(encoder_name=a.encoder, encoder_weights="imagenet",
                     in_channels=3, classes=1).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[训练] U-Net/{a.encoder}  参数量 {n_par/1e6:.1f} M")

    crit = ComboLoss().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    log, best = [], -1.0
    # 逐 epoch 追加写日志，方便外部实时监控（上次卡死时因为只在结尾写，完全看不到进度）
    log_path = OUT / "train_log.csv"
    with log_path.open("w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=["epoch", "train_loss", "dice", "iou",
                                            "precision", "recall", "lr", "sec", "peak_GPU_GB"])
        wr.writeheader()
        fh.flush()

        for ep in range(1, a.epochs + 1):
            model.train()
            t0, tot, n = time.time(), 0.0, 0
            for x, y in tr_ld:
                x = x.to(dev, non_blocking=True)
                y = y.to(dev, non_blocking=True).unsqueeze(1)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(x)
                    if logits.shape[-2:] != y.shape[-2:]:
                        logits = nn.functional.interpolate(logits, size=y.shape[-2:],
                                                           mode="bilinear", align_corners=False)
                    loss = crit(logits, y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                tot += float(loss.detach())
                n += 1
            sched.step()
            vm = evaluate(model, va_ld, dev)
            mem = torch.cuda.max_memory_allocated() / 1024**3 if dev.type == "cuda" else 0
            rec = {"epoch": ep, "train_loss": round(tot / max(1, n), 4),
                   "dice": round(vm["dice"], 4), "iou": round(vm["iou"], 4),
                   "precision": round(vm["precision"], 4), "recall": round(vm["recall"], 4),
                   "lr": round(sched.get_last_lr()[0], 6),
                   "sec": round(time.time() - t0, 1), "peak_GPU_GB": round(mem, 2)}
            log.append(rec)
            wr.writerow(rec)
            fh.flush()
            star = ""
            if vm["dice"] > best:
                best = vm["dice"]
                torch.save({"model": model.state_dict(), "encoder": a.encoder, "epoch": ep,
                            "val": vm, "args": vars(a)}, OUT / "best_unet.pt")
                star = "  * best"
            print(f"  ep{ep:3d}/{a.epochs}  loss {rec['train_loss']:.4f}  "
                  f"dice {vm['dice']:.4f}  iou {vm['iou']:.4f}  "
                  f"prec {vm['precision']:.4f}  rec {vm['recall']:.4f}  "
                  f"{rec['sec']}s  GPU {mem:.2f}GB{star}", flush=True)

    (OUT / "train_summary.json").write_text(json.dumps(
        {"best_val_dice": best, "epochs": a.epochs, "final": log[-1],
         "n_train_patch": int(Xtr.shape[0]), "n_val_patch": int(Xva.shape[0]),
         "encoder": a.encoder, "seed": SEED}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  最佳 val Dice {best:.4f}   权重 -> {OUT/'best_unet.pt'}")

    # ---- 曲线 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = [r["epoch"] for r in log]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(ep, [r["train_loss"] for r in log]); ax[0].set_title("train loss")
    ax[1].plot(ep, [r["dice"] for r in log], label="dice")
    ax[1].plot(ep, [r["iou"] for r in log], label="iou"); ax[1].legend(); ax[1].set_title("val")
    ax[2].plot(ep, [r["precision"] for r in log], label="prec")
    ax[2].plot(ep, [r["recall"] for r in log], label="rec"); ax[2].legend()
    ax[2].set_title("val precision / recall")
    for x in ax:
        x.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(OUT / "training_curve.png", dpi=110)
    print(f"  曲线 -> {OUT/'training_curve.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
