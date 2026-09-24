"""生成"抗锯齿+接断点"版（auto3），供送 WinRHIZO 对照。

对比对象:
  auto  (收边1px, 无平滑)       -> 根总长 +11.0%  根尖 +54.2%
  auto2 (收边2px, 无平滑)       -> 根总长  +7.2%  根尖 +88.0%
  auto3 (收边1px + 抗锯齿接断点) -> 本次

输出: 测试\\7-003_auto3.tif + 项目 outputs 同名一份
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import rootseg_common as R  # noqa: E402

# --- 统一路径（见 paths.py；数据搬家只改那里）---
sys.path.insert(0, str(Path(__file__).parent))
import paths as P  # noqa: E402

spec_src = (Path(__file__).parent / "38_export_tiff.py").read_text(encoding="utf-8")
ns = {"__file__": str(Path(__file__).parent / "38_export_tiff.py")}
exec(compile(spec_src.split("def main(")[0], "exp", "exec"), ns)
remove_loops = ns["remove_loops"]

Image.MAX_IMAGE_PIXELS = None
PROJ = P.PROJ
ORIG = P.RAW_D2_99 / "7-003.tif"
OUTDIRS = [PROJ / "outputs" / "7-003三方对比", P.TEST]


def build(tag, shrink, smooth_open, bridge, halo=0):
    R.HALO_GROW = halo
    R.SHRINK_PX = shrink
    R.SMOOTH_ALL = (smooth_open > 0 or bridge > 0)
    R.SMOOTH_OPEN_PX = smooth_open
    R.BRIDGE_CLOSE_PX = bridge
    orig = np.asarray(Image.open(ORIG).convert("RGB")).copy()
    root, _, _ = R.extract_root(orig, return_maps=False)
    out = orig.copy()
    out[~root] = 255
    out[remove_loops(out)] = 255
    m = np.any(out < 250, axis=2)
    img = Image.fromarray(out)
    for d in OUTDIRS:
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"7-003_{tag}.tif"
        img.save(p, format="TIFF", compression="raw", dpi=(600, 600))
    print(f"  {tag:<8} 收边{shrink}px 开{smooth_open} 闭{bridge}  ->  前景 {100*m.mean():.3f}%  "
          f"{os.path.getsize(OUTDIRS[0]/f'7-003_{tag}.tif')/1024**2:.1f} MB")
    return out


if __name__ == "__main__":
    print("生成中 …")
    build("auto3", 1, 1, 2)      # 收边1px + 抗锯齿 + 接断点
    build("auto4", 2, 1, 2)      # 收边2px + 抗锯齿 + 接断点
    print("完成")
