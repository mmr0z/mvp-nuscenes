#!/usr/bin/env bash
MVP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${MVP_ROOT}:${MVP_ROOT}/third_party:${MVP_ROOT}/third_party/CenterPoint:${MVP_ROOT}/third_party/CenterNet2:${MVP_ROOT}/third_party/CenterNet2/projects/CenterNet2:${MVP_ROOT}/third_party/detectron2:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MVP_ROOT}/.cache/matplotlib"
mkdir -p "$MPLCONFIGDIR"
