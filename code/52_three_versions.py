"""三版本对照盘点：找出同时拥有 人工 / CV / U-Net 三个版本的样本编号。

版本定义（按用户 2026-09-23 说明）:
  人工  : `测试\\人工\\9.9改\\<样本>-man*.tif`（从原图手工修）
          或 `人工修复mac图\\<样本>-unet-man.tif`（修 U-Net 候选得到的）
  CV    : `白底根图\\<批次>\\<样本>.jpg` 或 `白底根图_第二数据集\\<批次>\\<样本>.jpg`
          （经典规则管线 rootseg_common 的产出）
  U-Net : `outputs\\预标注候选\\<样本>-mac.tif`（49_preannotate.py 的产出）

同时检查每个文件的**可打开性**（用户反馈"剩下几张图打不开"）。
产出: outputs/三版本盘点/three_versions.csv
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import paths as P  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
OUT = P.PROJ / "outputs" / "三版本盘点"
OUT.mkdir(parents=True, exist_ok=True)


def probe(p: Path) -> tuple[bool, str]:
    """返回 (能否打开, 说明)"""
    if not p.exists():
        return False, "不存在"
    try:
        with Image.open(p) as im:
            im.load()          # 真正解码，不只是读头
            return True, f"{im.size[0]}x{im.size[1]} {im.mode}"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def index(dirp: Path, suffixes) -> dict[str, Path]:
    """按后缀建索引。suffixes 里给空串 "" 表示"文件名本身就是样本号"（不剥后缀）。"""
    m = {}
    if not dirp.exists():
        return m
    for p in dirp.rglob("*"):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".tif", ".tiff", ".png"):
            continue
        s = p.stem
        for suf in suffixes:
            if suf == "":
                m[s] = p          # 无后缀：直接用 stem 做 key
                break
            if s.endswith(suf):
                m[s[: -len(suf)]] = p
                break
    return m


def main() -> int:
    # U-Net 产物（可能已被改名成 -unet）
    cand = P.PROJ / "outputs" / "预标注候选"
    unet = index(cand, ["-unet", "-mac"])
    unet = {k: v for k, v in unet.items() if v.parent == cand}

    # 人工
    man = {}
    for d in [P.TEST_MAN_99GAI, P.TEST_MAN, P.PROJ / "人工修复mac图"]:
        man.update(index(d, ["-unet-man", "-man"]))

    # CV
    cv = {}
    for d in [P.WHITE_D1, P.WHITE_D2]:
        cv.update(index(d, [""]))

    all_ids = sorted(set(unet) | set(man) | set(cv))
    print(f"U-Net {len(unet)}   人工 {len(man)}   CV {len(cv)}   并集 {len(all_ids)}\n")

    both3 = [k for k in all_ids if k in unet and k in man and k in cv]
    print(f"=== 同时有 人工 + CV + U-Net 三版的样本: {len(both3)} 个 ===")
    for k in both3:
        print(f"  {k}")

    print(f"\n=== 有人工 + U-Net（缺 CV）===")
    for k in all_ids:
        if k in man and k in unet and k not in cv:
            print(f"  {k}")

    rows = []
    bad = []
    for k in all_ids:
        rec = {"样本": k,
               "人工": str(man.get(k, "")), "CV": str(cv.get(k, "")), "UNet": str(unet.get(k, ""))}
        for tag, path in (("人工", man.get(k)), ("CV", cv.get(k)), ("UNet", unet.get(k))):
            if path is None:
                rec[f"{tag}_状态"] = "无"
                continue
            ok, msg = probe(path)
            rec[f"{tag}_状态"] = ("OK " if ok else "坏 ") + msg
            if not ok:
                bad.append((k, tag, str(path), msg))
        rows.append(rec)

    with (OUT / "three_versions.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    print(f"\n=== 打不开 / 异常的文件: {len(bad)} 个 ===")
    for k, tag, path, msg in bad[:40]:
        print(f"  [{k}] {tag}: {msg}")
        print(f"        {path}")

    print(f"\n  清单 -> {OUT/'three_versions.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
