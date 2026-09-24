"""阶段二 · 预标注方案可行性探针。

背景: 任务书要求先试 SAM2。实测结论——SAM2 在 PyPI 无 Windows 预编译轮子
      (pip install sam2 --only-binary=:all: -> No matching distribution found)，
      且本机没有 MSVC 编译工具(vswhere 查无 Visual Studio Build Tools, C++ 工具链缺失)，
      CUDA 算子无法编译 => 按任务书"装不上就跳过，不要卡在这一步"。

本脚本检查两个**免编译**的替代预标注方案是否可用:
      1. segment-anything (SAM v1, 纯 PyTorch 实现，不需要编译 CUDA 算子)
      2. mobile-sam (轻量 SAM，权重约 40MB)
      另附一个零依赖方案: 阶段一已有的经典流水线(照度校正 + bottom-hat + Frangi)
"""
from __future__ import annotations

import importlib
import json
import sys


def check(mod_name: str) -> dict:
    try:
        m = importlib.import_module(mod_name)
        has_hq = hasattr(m, "sam_model_registry") or hasattr(m, "sam_model_registry")
        return {"ok": True, "version": str(getattr(m, "__version__", "?")),
                "has_registry": bool(has_hq)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    report = {
        "sam2": {"ok": False,
                 "reason": "PyPI 无 Windows 预编译轮子；本机无 MSVC C++ 工具链，CUDA 算子无法编译"},
        "segment_anything": check("segment_anything"),
        "mobile_sam": check("mobile_sam"),
        "classic_pipeline": {"ok": True,
                             "reason": "零依赖，阶段一已实现并实测：根召回 90.3%，阴影误判为根 16.6%"},
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
