"""检查三张待清理图（22-001 / 44-001 / 46-002）的边框形态，为自动涂白定策略。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import paths as P  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
OUT = P.PROJ / "outputs" / "边框清理"
OUT.mkdir(parents=True, exist_ok=True)
NONWHITE = 250


def main() -> int:
    targets = ["22-001", "44-001", "46-002"]
    for stem in targets:
        p = None
        for d in [P.TEST_MAN_99GAI, P.TEST_MAN]:
            for q in d.glob("*"):
                if "-man" in q.stem.lower() and \
                        q.stem.replace("-man", "").replace("-9.22", "") == stem:
                    p = q
        if p is None:
            print(f"  {stem}: 找不到")
            continue
        a = np.asarray(Image.open(p).convert("RGB"))
        H, W = a.shape[:2]
        nw = np.any(a < NONWHITE, axis=2)
        g = np.asarray(Image.open(p).convert("L")).astype(np.float32)

        print(f"\n=== {stem}  ({W}x{H})  全图非白 {100*nw.mean():.2f}% ===")
        # 逐行/逐列的"贯通占比"
        rows = nw.mean(axis=1)
        cols = nw.mean(axis=0)
        print("  从上往下 贯通占比（前 12 档，每档 40 行）:")
        print("   ", "  ".join(f"{100*rows[i*40:(i+1)*40].mean():5.1f}%" for i in range(12)))
        print("  从下往上（前 12 档）:")
        print("   ", "  ".join(f"{100*rows[H-(i+1)*40:H-i*40].mean():5.1f}%" for i in range(12)))
        print("  从左往右（前 12 档，每档 40 列）:")
        print("   ", "  ".join(f"{100*cols[i*40:(i+1)*40].mean():5.1f}%" for i in range(12)))
        print("  从右往左（前 12 档）:")
        print("   ", "  ".join(f"{100*cols[W-(i+1)*40:W-i*40].mean():5.1f}%" for i in range(12)))
        # 边框区灰度
        b = np.zeros_like(nw)
        b[:80, :] = b[-80:, :] = b[:, :80] = b[:, -80:] = True
        print(f"  边框 80px 带: 非白 {100*nw[b].mean():.1f}%  "
              f"其中非白像素灰度中位 {np.median(g[b & nw]) if (b&nw).any() else -1:.0f}")

        # 存预览
        im = Image.open(p).convert("RGB")
        k = 1100 / im.width
        im.resize((1100, int(im.height * k)), Image.LANCZOS).save(
            OUT / f"{stem}_preview.jpg", quality=88)
        # 左上角 1:1
        im.crop((0, 0, 900, 700)).save(OUT / f"{stem}_corner_TL.png")
        im.crop((im.width - 900, im.height - 700, im.width, im.height)).save(
            OUT / f"{stem}_corner_BR.png")
    print(f"\n  预览 -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
