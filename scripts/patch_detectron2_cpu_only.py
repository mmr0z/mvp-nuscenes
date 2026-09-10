from __future__ import annotations

import sys
from pathlib import Path


OLD = """    extension = CppExtension

    extra_compile_args = {\"cxx\": []}
    define_macros = []

    if (torch.cuda.is_available() and ((CUDA_HOME is not None) or is_rocm_pytorch)) or os.getenv(
        \"FORCE_CUDA\", \"0\"
    ) == \"1\":
"""

NEW = """    extension = CppExtension

    cpu_only = os.getenv(\"D2_CPU_ONLY\", \"0\") == \"1\"
    extra_compile_args = {\"cxx\": []}
    define_macros = []

    if not cpu_only and ((torch.cuda.is_available() and ((CUDA_HOME is not None) or is_rocm_pytorch)) or os.getenv(
        \"FORCE_CUDA\", \"0\"
    ) == \"1\"):
"""


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: patch_detectron2_cpu_only.py /path/to/detectron2", file=sys.stderr)
        return 2

    repo_dir = Path(sys.argv[1]).resolve()
    setup_py = repo_dir / "setup.py"

    if not setup_py.is_file():
        print(f"setup.py not found: {setup_py}", file=sys.stderr)
        return 1

    content = setup_py.read_text()
    if "cpu_only = os.getenv(\"D2_CPU_ONLY\", \"0\") == \"1\"" in content:
        return 0

    if OLD not in content:
        print("expected detectron2 setup.py block not found", file=sys.stderr)
        return 1

    setup_py.write_text(content.replace(OLD, NEW, 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
