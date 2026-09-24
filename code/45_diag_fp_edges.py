"""诊断误检的空间分布：误检是不是集中在图像边缘？

观察: pred_66-003 的误差图里，红色（误检）几乎全在四周边缘带。
本脚本按"到边缘的距离"分环统计误检占比，确认这一点。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import paths as P  # noqa: E402
import importlib.util as _il

spec = _il.spec_from_file_location("inf", str(Path(__file__).parent / "44_infer_unet.py"))

Image.MAX_IMAGE_PIXELS = None
OUT = P.PROJ / "outputs" / "训练"
NONWHITE = 250
RINGS = [(0, 60), (60, 150), (150, 300), (300, 600), (600, 10 ** 9)]


def main() -> int:
    # 复用 44 的函数（不执行其 main）
    src = (Path(__file__).parent / "44_infer_unet.py").read_text(encoding="utf-8")
    ns = {"__name__": "inf", "__file__": str(Path(__file__).parent / "44_infer_unet.py")}
    exec(compile(src.split("def main(")[0], "inf", "exec"), ns)
    infer_full, find_orig, find_man = ns["infer_full"], ns["find_orig"], ns["find_man"]

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(OUT / "best_unet.pt", map_location="cpu", weights_only=False)
    import segmentation_models_pytorch as smp
    model = smp.Unet(encoder_name=ck["encoder"], encoder_weights=None,
                     in_channels=3, classes=1).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()

    import json
    val_ids = json.loads((OUT / "dataset_report.json").read_text(encoding="utf-8"))["val_ids"]

    print(f"{'样本':<12}{'误检总数':>10}  " + "  ".join(f"{a}-{b}px" if b < 10**9 else f">{a}px"
                                                        for a, b in RINGS))
    for stem in val_ids:
        o, m = find_orig(stem), find_man(stem)
        img = np.asarray(Image.open(o).convert("RGB"))
        gt = np.any(np.asarray(Image.open(m).convert("RGB")) < NONWHITE, axis=2)
        pred = infer_full(model, img, dev) > 0.5
        fp = (~gt) & pred
        H, W = gt.shape
        yy, xx = np.mgrid[0:H, 0:W]
        edge_dist = np.minimum(np.minimum(yy, H - 1 - yy), np.minimum(xx, W - 1 - xx))
        tot = int(fp.sum())
        parts = []
        for a, b in RINGS:
            n = int((fp & (edge_dist >= a) & (edge_dist < b)).sum())
            parts.append(f"{100*n/max(1,tot):5.1f}%")
        print(f"{stem:<12}{tot:>10}  " + "  ".join(parts))

        # 顺带报告：把边缘带整条判为背景后，指标改善多少
        for trim in (150, 300):
            inside = edge_dist >= trim
            g, p2 = gt & inside, pred & inside
            inter = float((g & p2).sum())
            dice = 2 * inter / max(1e-6, g.sum() + p2.sum())
            iou = inter / max(1e-6, (g | p2).sum())
            prec = inter / max(1e-6, p2.sum())
            rec = inter / max(1e-6, g.sum())
            print(f"    {'裁边'+str(trim)+'px后':<12} Dice {dice:.4f}  IoU {iou:.4f}  "
                  f"精确 {prec:.4f}  召回 {rec:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
