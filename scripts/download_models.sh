#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
gh release download models-v1 --repo "${1:-mmr0z/mvp-training}" --pattern '*.pth' --dir models
python3 scripts/verify_models.py
