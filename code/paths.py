"""项目统一路径管理 —— 数据搬家只改这一个文件。

用法:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    import paths as P

    files = sorted(P.RAW_D2_99.glob("*.tif"))

变更记录:
  2026-09-23  用户把 `桌面\\采样数据\\图片` 移到 `扫描\\原始图片`；
              并把 `多样性林\\9.9改`（手工在制品）移到 `测试\\人工\\9.9改`。
"""
from __future__ import annotations

from pathlib import Path

# 由当前用户目录推导，不写死用户名；换机器/换用户名都不用改这里。
DESKTOP = Path.home() / "Desktop"
# 注意: DESKTOP 只用于"仍在桌面的外部数据"（原始图 SCAN / 测试区 TEST）。
#       项目本体 PROJ 已搬到 D 盘，不再由 DESKTOP 推导。

# ---------- 项目本体（2026-09-23 整体搬到 D 盘；C 盘只剩 17GB）----------
PROJ = Path(r"D:\根系分割项目")
CODE = PROJ / "code"
DOCS = PROJ / "docs"
CONFIG = PROJ / "config"
MODELS = PROJ / "models"
LABELS = PROJ / "labels"
OUTPUTS = PROJ / "outputs"

# ---------- 原始扫描图 ----------
SCAN = DESKTOP / "扫描"
RAW_D1 = SCAN / "图片"                    # 数据集1（108 张，纵向 4962x7019）
RAW_D2 = SCAN / "原始图片"                 # 数据集2（215 张，横向 7019x4962）
RAW_D2_99 = RAW_D2 / "多样性林" / "9.9"     # 数据集2 的 9.9 子集

# ---------- 用户的测试 / 真值区 ----------
TEST = DESKTOP / "测试"
TEST_ORIG = TEST / "原图"
TEST_MAN = TEST / "人工"
TEST_MAC = TEST / "机器"
TEST_DBG = TEST / "调试"
TEST_MAN_99GAI = TEST / "人工" / "9.9改"        # 用户手工修图（在制品，只读）
TEST_MAC_99GAI = TEST / "机器" / "9.9改mac"     # 对应的机器图输出
TEST_DATA = TEST / "测试数据.TXT"               # WinRHIZO 输出（GBK，制表符分隔）

# ---------- 项目内产物 ----------
PROCESSED = PROJ / "processed"
PROCESSED_D1 = PROCESSED / "600dpi"            # 数据集1 预处理后（已裁边）
PROCESSED_D1_EXCL = PROCESSED / "600dpi_excluded"
WHITE_D1 = PROJ / "白底根图"                    # 数据集1 白底根图
WHITE_D2 = PROJ / "白底根图_第二数据集"            # 数据集2 白底根图

# ---------- 旧路径（仅供历史脚本引用，勿用于新代码）----------
OLD_SRC2 = DESKTOP / "采样数据" / "图片"


def check() -> None:
    """打印关键路径是否存在，便于排查。"""
    for name in ["PROJ", "RAW_D1", "RAW_D2", "RAW_D2_99", "TEST", "TEST_MAN_99GAI",
                 "TEST_MAC_99GAI", "PROCESSED_D1", "WHITE_D1", "WHITE_D2", "TEST_DATA"]:
        p = globals()[name]
        print(f"  {'OK ' if p.exists() else '缺失'} {name:<16} {p}")


if __name__ == "__main__":
    check()
