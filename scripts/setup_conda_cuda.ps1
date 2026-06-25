param(
    [string]$EnvName = "kan-cgcnn-cuda"
)

$ErrorActionPreference = "Stop"

conda env create -n $EnvName -f environment-cuda.yml
conda run -n $EnvName python -m pip install matbench==0.6 --no-deps
conda run -n $EnvName python -c "import torch, torch_geometric, matbench; print('torch:', torch.__version__); print('torch_geometric:', torch_geometric.__version__); print('matbench:', matbench.__version__); print('cuda_available:', torch.cuda.is_available()); print('cuda_device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
