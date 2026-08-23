# Local Windows + RTX 40-series + VS Code Jupyter

This is the fastest local path for the final experiments when Colab is unavailable.

Tested target setup for this repository:

- Windows 10/11 x64
- NVIDIA RTX 40-series GPU (the project fits in 16 GB VRAM on GPT-2 Small)
- Python 3.11 x64
- PyTorch 2.10.0 CUDA 12.8 wheel
- VS Code with the **Python** and **Jupyter** extensions

A separate CUDA Toolkit installation is not required for the pip PyTorch wheel. You do need a recent NVIDIA driver. Run `nvidia-smi` first; update the driver if it cannot see the GPU.

## 1. Pull the latest repository

PowerShell, from the repository root:

```powershell
git pull --ff-only
```

## 2. Create a clean Python 3.11 environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
```

Using the explicit `.venv` Python path avoids PowerShell execution-policy problems; activation is optional.

## 3. Install CUDA-enabled PyTorch first

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
```

Verify immediately:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print('torch',torch.__version__); print('cuda build',torch.version.cuda); print('cuda available',torch.cuda.is_available()); print('gpu',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'); print('vram GB',round(torch.cuda.get_device_properties(0).total_memory/1024**3,2) if torch.cuda.is_available() else 0)"
```

Do not continue if `cuda available` is `False`.

## 4. Install all project + notebook dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-local-windows.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

`requirements-local-windows.txt` includes the project requirements plus:

- `ipykernel`
- `ipywidgets`
- `tabulate`
- `psutil`

The base project dependencies include PyTorch, TransformerLens, Transformers, Datasets, NumPy, pandas, matplotlib, PyYAML, tqdm, pytest, and blobfile.

## 5. Register the VS Code notebook kernel

```powershell
.\.venv\Scripts\python.exe -m ipykernel install --user --name steering-repair --display-name "Python (steering-repair CUDA)"
```

In VS Code:

1. Open the repository folder.
2. Install the Microsoft **Python** and **Jupyter** extensions if necessary.
3. Open `notebooks/selective_jrr_vscode_windows.ipynb`.
4. In the top-right kernel selector choose **Python (steering-repair CUDA)**.
5. Run cells from top to bottom.

## 6. Optional Hugging Face token

The models are public, so a token is not required. It can reduce Hub rate-limit problems.

PowerShell for the current process:

```powershell
$env:HF_TOKEN="hf_..."
```

Do not commit a token to the repository or notebook.

## 7. Manual final-experiment commands

The local notebook runs these for you, but the equivalent commands are:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_selective_jrr.py tests/test_jrr.py

.\.venv\Scripts\python.exe scripts/preflight_selective_jrr.py --config configs/selective_jrr_gpt2.yaml

.\.venv\Scripts\python.exe scripts/run_selective_jrr.py --config configs/selective_jrr_gpt2.yaml --phase calibration
```

Only if `results/selective_jrr/calibration_summary.json` contains `"go_to_new_heldout": true`:

```powershell
.\.venv\Scripts\python.exe scripts/run_selective_jrr.py --config configs/selective_jrr_gpt2.yaml --phase evaluation
```

Do not use `--force` to bypass the calibration gate.

## 8. If the frozen sentiment direction is missing

The notebook checks this automatically. Manually:

```powershell
.\.venv\Scripts\python.exe scripts/validate_sentiment_baseline.py --config configs/baseline_sentiment_gpt2.yaml
```

This should create `results/sentiment_direction.pt`.

## 9. Common Windows failures

### `torch.cuda.is_available() == False`

Usually the wrong CPU PyTorch build or an old NVIDIA driver. Reinstall the CUDA wheel explicitly:

```powershell
.\.venv\Scripts\python.exe -m pip uninstall -y torch
.\.venv\Scripts\python.exe -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
```

### VS Code uses another Python

In the notebook kernel selector choose **Python (steering-repair CUDA)**. Then run:

```python
import sys
print(sys.executable)
```

It must point to this repository's `.venv\\Scripts\\python.exe`.

### `No module named steering_repair`

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
```

### `Missing optional dependency 'tabulate'`

```powershell
.\.venv\Scripts\python.exe -m pip install tabulate
```

### Hugging Face download warnings / rate limits

Set `HF_TOKEN` as above and restart the notebook kernel so the environment variable is visible.

## 10. What to send back

At the end, zip `results/selective_jrr/`. The local notebook contains a final cell that writes `selective_jrr_results_local.zip` in the repository root.
