"""项目就绪度盘点：环境 / 数据 / 标签 / 训练脚本"""
from __future__ import annotations

import importlib
import json
from pathlib import Path

PROJ = Path(r"D:\根系分割项目")

print("=" * 60)
print("1. 环境 / 算力")
print("=" * 60)
try:
    import torch
    print(f"  torch            {torch.__version__}   cuda_build={torch.version.cuda}")
    print(f"  cuda_available   {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"  GPU              {p.name}  sm_{p.major}{p.minor}  "
              f"{p.total_memory/1024**3:.1f} GB")
        x = torch.randn(2, 3, 64, 64, device="cuda")
        conv = torch.nn.Conv2d(3, 8, 3, padding=1).cuda()
        y = conv(x); y.pow(2).mean().backward()
        print(f"  冒烟测试          forward {list(y.shape)}  backward OK")
except Exception as e:  # noqa: BLE001
    print(f"  torch 异常: {type(e).__name__}: {e}")

for m in ["segmentation_models_pytorch", "albumentations", "timm", "cv2", "skimage", "numpy"]:
    try:
        mod = importlib.import_module(m)
        v = getattr(mod, "__version__", "?")
        print(f"  {m:32s} {v}")
    except Exception:  # noqa: BLE001
        print(f"  {m:32s} 缺失")

print()
print("=" * 60)
print("2. 图像数据")
print("=" * 60)
for rel in ["processed/600dpi", "白底根图", "白底根图_第二数据集"]:
    d = PROJ / rel
    fs = [p for p in d.rglob("*") if p.suffix.lower() in (".jpg", ".png", ".tif")]
    mb = sum(p.stat().st_size for p in fs) / 1024**2
    print(f"  {rel:30s} {len(fs):5d} 张  {mb/1024:7.2f} GB")

print()
print("=" * 60)
print("3. 标签 / 真值（训练的关键）")
print("=" * 60)
for rel in ["labels", "labels/mask_v0_candidate", "labels/mask_v1"]:
    d = PROJ / rel
    if d.exists():
        fs = [p for p in d.rglob("*.png")]
        print(f"  {rel:30s} {len(fs)} 个 png")
    else:
        print(f"  {rel:30s} 不存在")
# mask 的取值分布（判断是几类）
import numpy as np
from PIL import Image
cand = sorted((PROJ / "labels").rglob("*_mask.png"))
if cand:
    p = cand[0]
    a = np.asarray(Image.open(p))
    vals, cnts = np.unique(a, return_counts=True)
    print(f"  样例 {p.name}: dtype={a.dtype} shape={a.shape} 取值={dict(zip(vals.tolist(), cnts.tolist()))}")

print()
print("=" * 60)
print("4. 人工真值（用户手工产出）")
print("=" * 60)
GT = Path.home() / "Desktop" / "测试" / "人工" / "9.9改"
if GT.exists():
    for p in sorted(GT.glob("*")):
        print(f"  {p.name:20s} {p.stat().st_size/1024**2:7.1f} MB  "
              f"{__import__('datetime').datetime.fromtimestamp(p.stat().st_mtime):%m-%d %H:%M}")
    print(f"  合计 {len(list(GT.glob('*')))} 张")

print()
print("=" * 60)
print("5. 训练脚本")
print("=" * 60)
code = PROJ / "code"
hits = [p.name for p in sorted(code.glob("*.py"))
        if any(k in p.name.lower() for k in ("train", "unet", "loss", "loader", "model"))]
print(f"  匹配 train/unet/loss/loader/model 的脚本: {hits if hits else '无'}")
print(f"  code 目录共 {len(list(code.glob('*.py')))} 个脚本")
print(f"  最近修改的 8 个: {[p.name for p in sorted(code.glob('*.py'), key=lambda z: -z.stat().st_mtime)[:8]]}")

print()
print("=" * 60)
print("6. 切分方案（train/val/test）")
print("=" * 60)
cfg = PROJ / "config"
for p in sorted(cfg.glob("*")):
    print(f"  config/{p.name}")
sp = cfg / "split.json"
print(f"  数据切分文件 split.json: {'存在' if sp.exists() else '不存在'}")
