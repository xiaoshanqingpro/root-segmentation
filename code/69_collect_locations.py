"""项目地址总收集：把所有相关位置、体量、内容列成一份清单。"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as P  # noqa: E402

Image = None

LOCATIONS = [
    ("项目本体", P.PROJ, "全部代码 / 产出 / 模型 / 文档"),
    ("├ 机器图_CV", P.PROJ / "机器图_CV", "经典规则管线产出（纯黑白二值）"),
    ("├ 机器图_UNet", P.PROJ / "机器图_UNet", "U-Net 产出（保留原色白底图）"),
    ("├ outputs", P.PROJ / "outputs", "中间产物 / 评估 / 对比图 / 训练"),
    ("├ processed", P.PROJ / "processed", "数据集1 预处理（裁边+600dpi）"),
    ("├ 白底根图", P.PROJ / "白底根图", "数据集1 旧版白底图"),
    ("├ 白底根图_第二数据集", P.PROJ / "白底根图_第二数据集", "数据集2 旧版白底图"),
    ("├ docs", P.PROJ / "docs", "全部文档"),
    ("├ code", P.PROJ / "code", "全部脚本"),
    ("├ labels", P.PROJ / "labels", "SAM 候选标签"),
    ("├ models", P.PROJ / "models", "SAM 权重"),
    ("└ _code_backup_20260923", P.PROJ / "_code_backup_20260923", "代码备份"),
    ("原始数据·数据集1", P.RAW_D1, "108 张，纵向 4962x7019"),
    ("原始数据·数据集2", P.RAW_D2, "215 张，横向 7019x4962"),
    ("原始数据·杉阔混交林", (Path.home() / "Desktop" / "杉阔混交林"), "151 张，纵横向混合"),
    ("真值/测试区", P.TEST, "人工标注、机器图、调试"),
    ("├ 人工", P.TEST_MAN, "人工标注（-man）"),
    ("├ 机器", P.TEST_MAC, "机器图（-mac）"),
    ("├ 原图", P.TEST_ORIG, "测试用原图"),
    ("└ 测试数据.TXT", P.TEST_DATA, "WinRHIZO 端到端输出"),
    ("旧位置·采样数据", (Path.home() / "Desktop" / "采样数据"), "图片已移走，剩 OCR 脚本"),
    ("残留·C盘根系分割项目", (Path.home() / "Desktop" / "扫描" / "根系分割项目"),
     "搬迁后残留（含另一线写的对比评估）"),
]


def du(path: Path) -> tuple[int, float]:
    n, b = 0, 0
    for root, dirs, files in os.walk(path):
        for f in files:
            try:
                b += os.path.getsize(os.path.join(root, f))
                n += 1
            except OSError:
                pass
    return n, b / 1024 ** 3


def main() -> int:
    print(f"{'位置':<26}{'文件':>7}{'GB':>8}  说明")
    print("-" * 96)
    total_n = total_b = 0
    for label, path, note in LOCATIONS:
        if not path.exists():
            print(f"{label:<26}{'—':>7}{'—':>8}  [缺失] {note}")
            continue
        if path.is_file():
            n, gb = 1, path.stat().st_size / 1024 ** 3
        else:
            n, gb = du(path)
            if label[:1] not in ("├", "└"):
                total_n += n
                total_b += gb
        print(f"{label:<26}{n:>7}{gb:>8.2f}  {note}")
    print("-" * 96)
    print(f"{'顶层合计':<26}{total_n:>7}{total_b:>8.2f}  （不含子项重复计数）")
    print()
    for d in ("C", "D"):
        try:
            u = __import__("shutil").disk_usage(f"{d}:\\")
            print(f"  {d}: 盘  已用 {u.used/1024**3:6.1f} GB   可用 {u.free/1024**3:6.1f} GB")
        except Exception:  # noqa: BLE001
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
