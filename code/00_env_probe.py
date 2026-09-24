"""阶段 0：环境探针。目标：确认 Python / GPU / 依赖可用性。
只读环境，不装任何包。输出 JSON 供后续脚本复用。"""
import importlib
import json
import platform
import subprocess
import sys

REQUIRED = [
    "numpy", "PIL", "cv2", "torch", "torchvision", "skimage",
    "scipy", "matplotlib", "albumentations", "tqdm", "pandas",
    "segmentation_models_pytorch",
]


def probe_packages():
    result = {}
    for name in REQUIRED:
        try:
            mod = importlib.import_module(name)
            result[name] = {"ok": True, "version": str(getattr(mod, "__version__", "?"))}
        except Exception as exc:  # noqa: BLE001
            result[name] = {"ok": False, "error": type(exc).__name__}
    return result


def probe_torch():
    info = {"installed": False}
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        info["error"] = f"{type(exc).__name__}: {exc}"
        return info

    info["installed"] = True
    info["version"] = torch.__version__
    info["cuda_build"] = torch.version.cuda
    info["cuda_available"] = torch.cuda.is_available()
    if info["cuda_available"]:
        info["device_name"] = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        info["capability"] = f"sm_{cap[0]}{cap[1]}"
        info["total_mem_GB"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 2
        )
        # 小卷积冒烟测试：前向 + 反向各一步
        try:
            torch.manual_seed(0)
            dev = torch.device("cuda:0")
            conv = torch.nn.Conv2d(3, 8, 3, padding=1).to(dev)
            x = torch.randn(2, 3, 64, 64, device=dev, requires_grad=False)
            y = conv(x)
            loss = y.pow(2).mean()
            loss.backward()
            grad_ok = conv.weight.grad is not None and torch.isfinite(conv.weight.grad).all().item()
            torch.cuda.synchronize()
            info["smoke_test"] = {
                "forward_shape": list(y.shape),
                "loss": round(float(loss.item()), 6),
                "backward_grad_ok": bool(grad_ok),
            }
        except Exception as exc:  # noqa: BLE001
            info["smoke_test"] = {"error": f"{type(exc).__name__}: {exc}"}
    return info


def probe_nvidia_smi():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30, check=False,
        )
        return out.stdout.strip() or out.stderr.strip()
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"


def main():
    report = {
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "bits": platform.architecture()[0],
        },
        "os": platform.platform(),
        "nvidia_smi": probe_nvidia_smi(),
        "torch": probe_torch(),
        "packages": probe_packages(),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    main()
