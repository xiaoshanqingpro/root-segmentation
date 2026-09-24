"""一次性补丁：把 code\\ 下所有脚本里的硬编码路径统一换成 paths.py 引用。

处理内容:
  1. 旧字面量 Path(r"C:\\Users\\LIHAOYANG\\Desktop\\采样数据\\图片...")  ->  P.XXX
  2. 重复定义的 PROJ / TEST 路径 -> P.PROJ / P.TEST
  3. 在 import 区插入 sys.path + import paths as P

跑完会打印每个文件的改动数；再跑一次应当全部为 0（幂等）。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

CODE = Path(r"D:\根系分割项目\code")
DESK = r"C:\Users\LIHAOYANG\Desktop"

# 顺序重要: 长的先替换，避免短前缀把长路径截断
RULES = [
    (rf'Path\(r"{re.escape(DESK)}\\采样数据\\图片\\多样性林\\9\.9改\\44-001\.tif"\)',
     'P.TEST_MAN_99GAI / "44-001.tif"'),
    (rf'Path\(r"{re.escape(DESK)}\\采样数据\\图片\\多样性林\\9\.9改\\7-003\.tif"\)',
     'P.TEST_MAN_99GAI / "7-003.tif"'),
    (rf'Path\(r"{re.escape(DESK)}\\采样数据\\图片\\多样性林\\9\.9\\7-003\.tif"\)',
     'P.RAW_D2_99 / "7-003.tif"'),
    (rf'Path\(r"{re.escape(DESK)}\\采样数据\\图片\\多样性林\\9\.9改"\)', 'P.TEST_MAN_99GAI'),
    (rf'Path\(r"{re.escape(DESK)}\\采样数据\\图片\\多样性林\\9\.9"\)', 'P.RAW_D2_99'),
    (rf'Path\(r"{re.escape(DESK)}\\采样数据\\图片"\)', 'P.RAW_D2'),
    (rf'Path\(r"{re.escape(DESK)}\\扫描\\图片"\)', 'P.RAW_D1'),
    (rf'Path\(r"{re.escape(DESK)}\\扫描\\原始图片"\)', 'P.RAW_D2'),
    (rf'Path\(r"{re.escape(DESK)}\\测试\\原图"\)', 'P.TEST_ORIG'),
    (rf'Path\(r"{re.escape(DESK)}\\测试\\人工"\)', 'P.TEST_MAN'),
    (rf'Path\(r"{re.escape(DESK)}\\测试\\机器"\)', 'P.TEST_MAC'),
    (rf'Path\(r"{re.escape(DESK)}\\测试\\测试数据\.TXT"\)', 'P.TEST_DATA'),
    (rf'Path\(r"{re.escape(DESK)}\\测试"\)', 'P.TEST'),
    (rf'Path\(r"{re.escape(DESK)}\\扫描\\根系分割项目"\)', 'P.PROJ'),
]

IMPORT_BLOCK = [
    "",
    "# --- 统一路径（见 paths.py；数据搬家只改那里）---",
    "sys.path.insert(0, str(Path(__file__).parent))",
    "import paths as P  # noqa: E402",
]


def has_import(src: str) -> bool:
    return re.search(r"^import paths as P\b", src, re.M) is not None


def insert_import(src: str) -> str:
    if has_import(src):
        return src
    lines = src.split("\n")
    # 找最后一条模块级 import / from
    last = -1
    for i, l in enumerate(lines):
        if re.match(r"^(import |from )\S", l) and "paths as P" not in l:
            last = i
    if last < 0:
        return src
    # 确保 sys 已导入
    if not re.search(r"^import sys\b", src, re.M):
        lines.insert(last + 1, "import sys")
        last += 1
    return "\n".join(lines[: last + 1] + IMPORT_BLOCK + lines[last + 1:])


def main() -> int:
    total = 0
    for p in sorted(CODE.glob("*.py")):
        if p.name == "paths.py":
            continue
        src = p.read_text(encoding="utf-8")
        new = src
        n = 0
        for pat, rep in RULES:
            new, k = re.subn(pat, rep, new)
            n += k
        # 旧的 PROJ/TEST 定义若还在，且文件里没有引用 P.*，保留不动（避免误伤）
        changed_import = False
        if n > 0 and not has_import(new):
            new = insert_import(new)
            changed_import = True
        if new != src:
            p.write_text(new, encoding="utf-8")
            total += n
            print(f"  {p.name:<34} 路径替换 {n:>2} 处"
                  f"{'  + 插入 paths 导入' if changed_import else ''}")
    print(f"\n  共替换 {total} 处")
    return 0


if __name__ == "__main__":
    sys.exit(main())
