#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-modnet-kan}"
TORCH_SPEC="${TORCH_SPEC:-torch==2.4.1+cu121}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

conda env create -n "${ENV_NAME}" -f environment-modnet-kan-unified.yml
conda run -n "${ENV_NAME}" python -m pip install --extra-index-url "${TORCH_INDEX_URL}" "${TORCH_SPEC}"
conda run -n "${ENV_NAME}" python -m pip install matbench==0.6 --no-deps
conda run -n "${ENV_NAME}" python - <<'PY'
import sys
import torch
import tensorflow
import modnet
import matbench

print("python:", sys.version)
print("torch:", torch.__version__)
print("tensorflow:", tensorflow.__version__)
print("modnet:", getattr(modnet, "__version__", "unknown"))
print("matbench:", matbench.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device:", torch.cuda.get_device_name(0))
PY
