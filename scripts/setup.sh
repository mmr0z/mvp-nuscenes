#!/usr/bin/env bash
set -euo pipefail
MVP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$MVP_DIR"
python -c 'import sys; assert sys.version_info[:2] == (3,10), "Use Python 3.10"'
command -v nvcc >/dev/null || { echo 'CUDA toolkit with nvcc is required.' >&2; exit 1; }
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation -e third_party/detectron2
(cd third_party/CenterPoint/det3d/ops/iou3d_nms && python setup.py build_ext --inplace)
source env.sh
python scripts/check_env.py
python -c 'from det3d.models import build_detector; from det3d.ops.iou3d_nms import iou3d_nms_cuda; print("CenterPoint imports OK")'
