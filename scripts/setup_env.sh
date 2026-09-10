#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
BASE_PYTHON="${BASE_PYTHON:-/home/workstation/miniconda3/envs/fusion/bin/python}"
THIRD_PARTY_DIR="${ROOT_DIR}/third_party"
CENTERNET2_DIR="${THIRD_PARTY_DIR}/CenterNet2"
CENTERPOINT_DIR="${THIRD_PARTY_DIR}/CenterPoint"
DETECTRON2_DIR="${THIRD_PARTY_DIR}/detectron2"
DETECTRON2_REPO_URL="${DETECTRON2_REPO_URL:-https://github.com/facebookresearch/detectron2.git}"

if [[ ! -x "${BASE_PYTHON}" ]]; then
  echo "Base Python not found: ${BASE_PYTHON}" >&2
  echo "Set BASE_PYTHON to a Python 3.8/3.9 interpreter with torch installed." >&2
  exit 1
fi

mkdir -p "${THIRD_PARTY_DIR}"

if [[ ! -d "${VENV_DIR}" ]]; then
  "${BASE_PYTHON}" -m venv --system-site-packages "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
mkdir -p "${ROOT_DIR}/.cache/matplotlib"
export MPLCONFIGDIR="${ROOT_DIR}/.cache/matplotlib"

python -m pip install -r "${ROOT_DIR}/requirements-mvp.txt"

if [[ ! -d "${CENTERNET2_DIR}" ]]; then
  git clone https://github.com/xingyizhou/CenterNet2.git "${CENTERNET2_DIR}"
fi

if [[ ! -d "${CENTERPOINT_DIR}" ]]; then
  git clone https://github.com/tianweiy/CenterPoint.git "${CENTERPOINT_DIR}"
fi

if [[ ! -d "${DETECTRON2_DIR}" ]]; then
  git clone "${DETECTRON2_REPO_URL}" "${DETECTRON2_DIR}"
fi

if [[ ! -e "${ROOT_DIR}/CenterNet2" ]]; then
  ln -s "${CENTERNET2_DIR}" "${ROOT_DIR}/CenterNet2"
fi

if [[ ! -e "${ROOT_DIR}/CenterPoint" ]]; then
  ln -s "${CENTERPOINT_DIR}" "${ROOT_DIR}/CenterPoint"
fi

if ! python -c "import detectron2" >/dev/null 2>&1; then
  if python - <<'PY' >/dev/null 2>&1
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
  then
    python -m pip install --no-build-isolation -e "${DETECTRON2_DIR}"
  else
    python "${ROOT_DIR}/scripts/patch_detectron2_cpu_only.py" "${DETECTRON2_DIR}"
    D2_CPU_ONLY=1 FORCE_CUDA=0 python -m pip install --no-build-isolation -e "${DETECTRON2_DIR}"
  fi
fi

cat > "${ROOT_DIR}/env.sh" <<EOF
#!/usr/bin/env bash
ROOT_DIR="${ROOT_DIR}"
VENV_DIR="${VENV_DIR}"
if [[ -n "\${CONDA_PREFIX:-}" ]]; then
  if [[ "\${CONDA_DEFAULT_ENV:-}" == "base" ]]; then
    echo "warning: base conda env is active; prefer a dedicated project env before sourcing env.sh" >&2
  fi
  if [[ -x "\${CONDA_PREFIX}/bin/nvcc" ]]; then
    export CUDA_HOME="\${CONDA_PREFIX}"
    export CUDA_PATH="\${CONDA_PREFIX}"
    case ":\${PATH}:" in
      *":\${CUDA_HOME}/bin:"*) ;;
      *) export PATH="\${CUDA_HOME}/bin:\${PATH}" ;;
    esac
    if [[ "\${MVP_EXPORT_LD_LIBRARY_PATH:-0}" == "1" ]]; then
      export LD_LIBRARY_PATH="\${CUDA_HOME}/lib:\${CUDA_HOME}/lib64:\${LD_LIBRARY_PATH:-}"
    fi
  fi
elif [[ -z "\${VIRTUAL_ENV:-}" && -d "\${VENV_DIR}" ]]; then
  source "\${VENV_DIR}/bin/activate"
fi
export PYTHONPATH="${ROOT_DIR}:${CENTERNET2_DIR}:${CENTERNET2_DIR}/projects/CenterNet2:${CENTERPOINT_DIR}:\${PYTHONPATH:-}"
export MPLCONFIGDIR="${ROOT_DIR}/.cache/matplotlib"
EOF

chmod +x "${ROOT_DIR}/env.sh"

echo "Environment prepared."
echo "Load it with: source ${ROOT_DIR}/env.sh"
