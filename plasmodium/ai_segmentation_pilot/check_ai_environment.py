from __future__ import annotations

import json
import platform


def main():
    result = {"python": platform.python_version()}
    try:
        import torch

        result.update(
            {
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "gpu_memory_gb": (
                    round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
                    if torch.cuda.is_available() else None
                ),
            }
        )
    except Exception as exc:
        result["torch_error"] = f"{type(exc).__name__}: {exc}"
    try:
        import sam2

        result["sam2"] = str(getattr(sam2, "__version__", "import_ok"))
    except Exception as exc:
        result["sam2_error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

