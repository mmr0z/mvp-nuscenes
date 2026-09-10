from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY = REPO_ROOT / "third_party"

for path in [
    REPO_ROOT,
    THIRD_PARTY,
    THIRD_PARTY / "CenterNet2",
    THIRD_PARTY / "CenterNet2" / "projects" / "CenterNet2",
    THIRD_PARTY / "CenterPoint",
]:
    if path.exists():
        sys.path.insert(0, str(path))


REQUIRED_MODULES = [
    "numpy",
    "cv2",
    "torch",
    "tqdm",
    "yaml",
    "pyquaternion",
    "nuscenes",
    "yacs",
    "fvcore",
    "omegaconf",
    "detectron2",
]


def check_module(name: str) -> tuple[bool, str]:
    try:
        module = importlib.import_module(name)
    except Exception as exc:  # pragma: no cover - diagnostic script
        return False, f"{type(exc).__name__}: {exc}"
    version = getattr(module, "__version__", "unknown")
    return True, version


def main() -> int:
    os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".cache" / "matplotlib"))
    print(f"python={sys.version.split()[0]}")
    print(f"python_executable={sys.executable}")
    print(f"PYTHONPATH={os.environ.get('PYTHONPATH', '')}")

    failures = 0
    for module_name in REQUIRED_MODULES:
        ok, detail = check_module(module_name)
        status = "OK" if ok else "MISSING"
        print(f"{module_name}: {status} ({detail})")
        failures += int(not ok)

    try:
        import torch

        print(f"torch_cuda={torch.version.cuda}")
        print(f"cuda_available={torch.cuda.is_available()}")
    except Exception:
        pass

    try:
        from CenterNet2.train_net import setup  # noqa: F401

        print("CenterNet2.train_net: OK")
    except Exception as exc:  # pragma: no cover - diagnostic script
        failures += 1
        print(f"CenterNet2.train_net: MISSING ({type(exc).__name__}: {exc})")

    if failures:
        print(f"environment_status=INCOMPLETE missing={failures}")
        return 1

    print("environment_status=READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
