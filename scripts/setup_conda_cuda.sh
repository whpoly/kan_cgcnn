#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-kan-cgcnn-cuda}"

conda env create -n "${ENV_NAME}" -f environment-cuda.yml
conda run -n "${ENV_NAME}" python -m pip install matbench==0.6 --no-deps
conda run -n "${ENV_NAME}" python - <<'PY'
import torch
import torch_geometric
import matbench
print("torch:", torch.__version__)
print("torch_geometric:", torch_geometric.__version__)
print("matbench:", matbench.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device:", torch.cuda.get_device_name(0))
PY
