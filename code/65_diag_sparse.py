"""诊断：稀疏样本在 residue 清理前后各剩多少像素，以及被判删的原因。

用法:
    python 65_diag_sparse.py <样本名...>
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import rootseg_common as R  # noqa: E402
import residue as RES  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

Image.MAX_IMAGE_PIXELS = None
SRC = Path.home() / "Desktop" / "杉阔混交林"
EXTS = (".jpg", ".jpeg", ".tif", ".tiff", ".png")


def set_params():
    R.FEAT_LONG = 3000
    R.HALO_GROW = 0
    R.SHRINK_PX = 1
    R.SMOOTH_ALL = True
    R.SMOOTH_OPEN_PX = 1
    R.BRIDGE_CLOSE_PX = 2


def comp_stats(binmask: np.ndarray) -> str:
    n, lab, st, _ = cv2.connectedComponentsWithStats(binmask.astype(np.uint8), 8)
    if n <= 1:
        return "0 块"
    areas = sorted((int(st[j, cv2.CC_STAT_AREA]) for j in range(1, n)), reverse=True)
    return f"{n-1} 块, 前5面积 {areas[:5]}, 总面积 {sum(areas)}"


def main(stems: list[str]) -> int:
    set_params()
    for s in stems:
        p = next((q for q in SRC.rglob("*") if q.stem == s and q.suffix.lower() in EXTS), None)
        if p is None:
            print(f"  !! 找不到 {s}")
            continue
        im = Image.open(p)
        dpi = float((im.info.get("dpi") or (600, 600))[0])
        rgb = np.asarray(im.convert("RGB"))
        if abs(dpi - 600.0) > 1:
            k = 600.0 / dpi
            rgb = np.asarray(Image.fromarray(rgb).resize(
                (max(1, round(rgb.shape[1] * k)), max(1, round(rgb.shape[0] * k))), Image.LANCZOS))

        root, _, info = R.extract_root(rgb, return_maps=False)
        pre = int(root.sum())
        img = np.where(root, 0, 255).astype(np.uint8)
        img2, se = RES.clean_residue(img)
        post = int((img2 < 128).sum())

        print(f"\n=== {s}  ({p.relative_to(SRC)})  {rgb.shape[1]}x{rgb.shape[0]}  {dpi:.0f}dpi ===")
        print(f"  清理前 根像素 {pre:>9}  ({pre/rgb.shape[0]/rgb.shape[1]*100:.4f}%)   {comp_stats(root)}")
        print(f"  清理后 根像素 {post:>9}  ({post/rgb.shape[0]/rgb.shape[1]*100:.4f}%)   {comp_stats(img2 < 128)}")
        print(f"  删除明细 {se}")
        print(f"  判根信息 {info}")

        # 存清理前 mask 预览（根=黑），用于判断"判根层检出的到底是不是根"
        OUTD = Path(r"D:\根系分割项目\outputs\杉阔复核")
        OUTD.mkdir(parents=True, exist_ok=True)
        pre_img = Image.fromarray(np.where(root, 0, 255).astype(np.uint8)).convert("RGB")
        w = 900
        pre_small = pre_img.resize((w, max(1, round(pre_img.height * w / pre_img.width))), Image.LANCZOS)
        pre_small.save(OUTD / f"{s}_清理前mask.png")
        print(f"  清理前 mask -> {OUTD / (s + '_清理前mask.png')}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["12-MZ-003", "23-B-GG-002", "16-CL-001"]))
