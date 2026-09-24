"""把 CV 与 U-Net 两套机器图分成两个顶层文件夹，**保持原有组织结构**。

重组前:
    D:\\根系分割项目\\机器图\\数据集1\\...            (CV, 数据集1+2)
    D:\\根系分割项目\\机器图\\数据集2\\...
    D:\\根系分割项目\\机器图_杉阔混交林\\{9.8校准,F,F补}\\    (CV, 杉阔)
    D:\\根系分割项目\\机器图_杉阔混交林_unet\\{9.8校准,F,F补}\\ (U-Net, 杉阔)
    D:\\根系分割项目\\outputs\\预标注候选\\*-unet.tif       (U-Net, 数据集1/2 部分)

重组后:
    D:\\根系分割项目\\机器图_CV\\{数据集1,数据集2,杉阔混交林}\\...
    D:\\根系分割项目\\机器图_UNet\\{数据集1,数据集2,杉阔混交林}\\...

原则:
    - 同盘 **移动**（瞬时重命名），不复制、不删任何图；只清理变成空的旧目录
    - **不改任何文件名**（杉阔 U-Net 沿用 -mac.tif，数据集1/2 U-Net 沿用 -unet.tif，
      差异在 README 里说明）
    - 幂等：目标已存在则跳过
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import paths as P  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

PROJ = P.PROJ
CV = PROJ / "机器图_CV"
UN = PROJ / "机器图_UNet"
SHANKUO_SUBS = ["9.8校准", "F", "F补"]
log: list[str] = []


def mv(src: Path, dst: Path) -> None:
    """移动；源不存在则跳过，目标已存在则跳过。"""
    if not src.exists():
        log.append(f"  跳过（源不存在） {src}")
        return
    if dst.exists():
        log.append(f"  跳过（目标已存在） {dst}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    log.append(f"  {src.relative_to(PROJ)}  ->  {dst.relative_to(PROJ)}")


def rmdir_if_empty(d: Path) -> None:
    if d.exists() and not any(d.iterdir()):
        d.rmdir()
        log.append(f"  删除空目录 {d.relative_to(PROJ)}")


def main() -> int:
    # ---- 0. 先建好"样本 -> CV 相对路径"映射（用于给数据集1/2 的 U-Net 找位置）----
    cv_rel: dict[str, str] = {}
    for base, prefix in [(PROJ / "机器图", ""), (PROJ / "机器图_杉阔混交林", "杉阔混交林")]:
        if not base.exists():
            continue
        for p in base.rglob("*-mac.tif"):
            stem = p.name[:-len("-mac.tif")]
            rel = p.relative_to(base)
            cv_rel.setdefault(stem, str(Path(prefix) / rel.parent) if prefix else str(rel.parent))
    print(f"[0] 已收录 {len(cv_rel)} 个 CV 样本的相对目录")

    # ---- 1. CV：数据集1/2 ----
    print("\n[1] CV 数据集1+2  ->  机器图_CV\\")
    for name in ["数据集1", "数据集2"]:
        mv(PROJ / "机器图" / name, CV / name)
    mv(PROJ / "机器图" / "README.md", CV / "README_数据集12.md")
    mv(PROJ / "机器图" / "_批量清单.csv", CV / "_批量清单_数据集12.csv")
    mv(PROJ / "机器图" / "_跳过清单.csv", CV / "_跳过清单_数据集12.csv")
    rmdir_if_empty(PROJ / "机器图")

    # ---- 2. CV：杉阔混交林 ----
    print("\n[2] CV 杉阔混交林  ->  机器图_CV\\杉阔混交林\\")
    for sub in SHANKUO_SUBS:
        mv(PROJ / "机器图_杉阔混交林" / sub, CV / "杉阔混交林" / sub)
    mv(PROJ / "机器图_杉阔混交林" / "_批量清单.csv", CV / "杉阔混交林" / "_批量清单.csv")
    rmdir_if_empty(PROJ / "机器图_杉阔混交林")

    # ---- 3. U-Net：杉阔混交林 ----
    print("\n[3] U-Net 杉阔混交林  ->  机器图_UNet\\杉阔混交林\\")
    for sub in SHANKUO_SUBS:
        mv(PROJ / "机器图_杉阔混交林_unet" / sub, UN / "杉阔混交林" / sub)
    mv(PROJ / "机器图_杉阔混交林_unet" / "batch_unet_report.csv",
       UN / "杉阔混交林" / "batch_unet_report.csv")
    mv(PROJ / "机器图_杉阔混交林_unet" / "duplicates.csv",
       UN / "杉阔混交林" / "duplicates.csv")
    rmdir_if_empty(PROJ / "机器图_杉阔混交林_unet")

    # ---- 4. U-Net：数据集1/2（来自 outputs\预标注候选）----
    print("\n[4] U-Net 数据集1/2  ->  机器图_UNet\\数据集1|数据集2\\")
    pre = PROJ / "outputs" / "预标注候选"
    tifs = sorted(pre.glob("*-unet.tif")) if pre.exists() else []
    n_ok = n_miss = 0
    for p in tifs:
        stem = p.name[:-len("-unet.tif")]
        sub = cv_rel.get(stem)
        if sub is None:
            n_miss += 1
            log.append(f"  找不到 CV 归属，留在原地: {p.name}")
            continue
        mv(p, UN / sub / p.name)
        n_ok += 1
    print(f"  移入 {n_ok} 张，未能归属 {n_miss} 张（留在 outputs\\预标注候选）")

    # ---- 5. 报告 ----
    print("\n" + "=" * 70)
    for line in log:
        print(line)
    print("=" * 70)
    for tag, root in [("机器图_CV", CV), ("机器图_UNet", UN)]:
        fs = list(root.rglob("*.tif")) if root.exists() else []
        total = sum(f.stat().st_size for f in fs)
        print(f"{tag:<12} {len(fs):>4} 张   {total/2**30:6.2f} GB   {root}")
    print(f"\nD 盘可用 {shutil.disk_usage('D:/').free/2**30:.1f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
