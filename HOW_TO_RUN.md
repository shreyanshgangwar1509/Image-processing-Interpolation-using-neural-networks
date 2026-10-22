# How to Run and Extend the Super-Resolution Benchmark

Welcome to the Image Super-Resolution (Interpolation) Benchmark project!

---

## 1. What This Project Is

This repository provides a modular, reproducible pipeline to compare image upscaling (super-resolution) methods.
It degrades clean High-Resolution (HR) ground-truth images into Low-Resolution (LR) inputs, upscales them using classical interpolation (nearest, bilinear, bicubic, lanczos) or neural networks (SRCNN, ESPCN, EDSR-lite), and evaluates quality using quantitative metrics (PSNR, SSIM, LPIPS).

### Directory Structure Overview
- `configs/`: YAML experiment configs inheriting from shared `_base_*.yaml` files.
- `src/`: Core Python package divided into `data`, `models`, `losses`, `metrics`, `training`, `evaluation`, and `utils`.
- `scripts/`: Executable CLI tools (`train.py`, `evaluate.py`, `run_batch.py`, `aggregate_results.py`, `make_dummy_data.py`, `check_data.py`).
- `tests/`: Automated unit tests executable with `pytest`.
- `notebooks/`: Jupyter notebooks for visual and statistical analysis of results (`results_analysis.ipynb`).
- `data/` & `results/`: Input datasets and output logs/metrics/samples (git-managed with `.gitkeep`).

---

## 2. First-Time Setup

### On the Author's Machine (CPU/Development)
#### Linux / macOS:
```bash
git clone <repo-url>
cd sr-project
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Windows (PowerShell):
```powershell
git clone <repo-url>
cd sr-project
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### On the Friend's Machine (GPU Execution)
To leverage CUDA acceleration, install the CUDA build of PyTorch following official instructions from [pytorch.org](https://pytorch.org):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Verify GPU accessibility with:
```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## 3. Smoke Test (Do This First, Takes <1 Minute)

Run these three commands to verify everything is set up correctly:

1. Generate synthetic dummy images:
```bash
python scripts/make_dummy_data.py
```

2. Run the quick smoke test experiment on CPU:
```bash
python scripts/train.py --config configs/smoke_test.yaml
```

3. Run unit tests:
```bash
pytest
```

### What a Successful Run Looks Like
After running the smoke test, the folder `results/smoke_test/` will be generated containing:
- `config_resolved.yaml`: Full resolved configuration used.
- `env.txt`: Environment diagnostic information.
- `log.txt`: Console output logs with timestamps.
- `train_log.csv`: Training loss and validation PSNR/SSIM history per epoch.
- `checkpoints/`: Model state files (`last.pth` and `best.pth`).
- `metrics.csv`: Per-image evaluation PSNR/SSIM scores and inference times.
- `summary.json`: Aggregated mean metrics per model and dataset.
- `samples/`: Visual comparison PNGs (`*_sr.png`, `*_hr.png`, `*_lr.png`).

---

## 4. Getting the Real Datasets

To run full benchmark comparisons, manually place standard HR image datasets under the `data/` directory:

Expected folder structure:
- `data/DIV2K_train_HR/`: DIV2K training set (800 HR images).
- `data/Set5/`: Set5 test set (5 HR images).
- `data/Set14/`: Set14 test/val set (14 HR images).
- `data/BSD100/`: BSD100 test set (100 HR images).
- `data/Urban100/`: Urban100 test set (100 HR images).

### Dataset Notes:
- Search online for dataset names (e.g., DIV2K official page at ETH Zurich, or Kaggle mirrors).
- Ensure dataset folders contain **only** HR image files (`.png`, `.jpg`, `.jpeg`, `.bmp`).
- **Never overlap training and test images.**
- Verify folder contents and image dimensions using `check_data.py`:
  ```bash
  python scripts/check_data.py data/Set5 data/Set14 data/DIV2K_train_HR
  ```
- *Tip for fast development:* Override training images count with `--set data.train.max_images=100` to train on a smaller subset.

---

## 5. Running Real Experiments

### Train or Evaluate a Single Config:
```bash
python scripts/train.py --config configs/srcnn_x4_clean.yaml
```

### Override Config Parameters from Command Line:
```bash
python scripts/train.py --config configs/srcnn_x4_clean.yaml --set training.epochs=3 --set data.loader.batch_size=8
```

### Run Batch Experiments Sequentially:
```bash
python scripts/run_batch.py configs/*.yaml
```

### Evaluate a Pre-Trained Checkpoint Only:
```bash
python scripts/evaluate.py --config configs/srcnn_x4_clean.yaml --checkpoint results/srcnn_x4_clean/checkpoints/best.pth
```

---

## 6. The Two-Person Workflow with GitHub

1. **Friend (GPU Machine)**:
   - Pulls latest configs and code from GitHub: `git pull`
   - Runs full experiment batch: `python scripts/run_batch.py configs/*.yaml`
   - Commits and pushes small result summary files back:
     ```bash
     git add results/
     git commit -m "Run x4 clean experiments"
     git push
     ```

2. **Author (CPU Machine)**:
   - Pulls updated result files: `git pull`
   - Aggregates all benchmark summaries into consolidated tables:
     ```bash
     python scripts/aggregate_results.py
     ```
   - Opens `notebooks/results_analysis.ipynb` in Jupyter to analyze performance.

### Git Tracking Rules:
- **Tracked in Git**: Code, YAML configs, `results/*.csv`, `results/*.json`, `results/*.txt`, sample PNGs.
- **Gitignored**: Large dataset files (`data/*`), model checkpoints (`*.pth`), virtual environment folders.
- *Sharing Checkpoints*: Share large `.pth` checkpoint files out-of-band via Google Drive or cloud storage if necessary.

---

## 7. Reading the Results

- `results/<experiment>/metrics.csv`: Detailed evaluation metrics for every image evaluated.
- `results/<experiment>/summary.json`: Mean metrics, parameter counts, and timing summaries.
- `results/summary.csv`: Aggregated CSV table containing all benchmark runs across experiments.
- `results/summary.md`: Auto-generated Markdown report formatted per dataset, sorted by PSNR.
- `notebooks/results_analysis.ipynb`: Interactive plots, pivot tables, and visual side-by-side sample comparison.

---

## 8. How to Extend the Project (Additions Only)

The project uses self-registering components and automatic module discovery.
**To add a new feature, you ONLY need to add a new file — NEVER edit existing code.**

### 1. Add a New Model (`src/models/my_model.py`)
```python
import torch
import torch.nn as nn
from src.models.registry import MODELS

@MODELS.register("my_model")
class MyModel(nn.Module):
    trainable: bool = True

    def __init__(self, scale: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.scale = scale
        self.conv = nn.Conv2d(3, 3 * scale * scale, kernel_size=3, padding=1)
        self.ps = nn.PixelShuffle(scale)

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        return self.ps(self.conv(lr))
```
*YAML config line:*
```yaml
model:
  name: my_model
  params: {hidden_dim: 64}
```

### 2. Add a New Degradation Step (`src/data/my_degradation.py`)
```python
import torch
from src.data.registry import DEGRADATIONS

@DEGRADATIONS.register("brightness_shift")
class BrightnessShift:
    def __init__(self, shift: float = 0.1) -> None:
        self.shift = shift

    def describe(self) -> str:
        return f"bright_shift_{self.shift}"

    def __call__(self, img: torch.Tensor, generator=None) -> torch.Tensor:
        return (img + self.shift).clamp(0.0, 1.0)
```
*YAML config line:*
```yaml
degradation:
  - name: bicubic_downscale
  - name: brightness_shift
    shift: 0.05
```

### 3. Add a New Metric (`src/metrics/my_metric.py`)
```python
import torch
from src.metrics.registry import METRICS

@METRICS.register("mae_metric")
class MAEMetric:
    def __init__(self, y_channel: bool = True, crop_border=None) -> None:
        pass

    def __call__(self, sr: torch.Tensor, hr: torch.Tensor, scale: int) -> float:
        return float(torch.mean(torch.abs(sr - hr)).item())
```
*YAML config line:*
```yaml
evaluation:
  metrics:
    - {name: mae_metric}
```

### 4. Add a New Loss (`src/losses/my_loss.py`)
```python
import torch
import torch.nn as nn
from src.losses.registry import LOSSES

@LOSSES.register("smooth_l1")
class SmoothL1LossModule(nn.Module):
    def __init__(self, beta: float = 1.0) -> None:
        super().__init__()
        self.fn = nn.SmoothL1Loss(beta=beta)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.fn(pred, target)
```
*YAML config line:*
```yaml
loss: {name: smooth_l1, params: {beta: 0.5}}
```

### 5. Add a New Dataset Type (`src/data/my_dataset.py`)
```python
from torch.utils.data import Dataset
from src.data.registry import DATASETS

@DATASETS.register("custom_dataset")
class CustomDataset(Dataset):
    def __init__(self, **kwargs) -> None:
        pass
    def __len__(self) -> int:
        return 0
    def __getitem__(self, idx: int):
        return {}
```

---

## 9. Troubleshooting

- **CUDA Out of Memory**: Lower `data.loader.batch_size` or `data.train.patch_size` in the YAML config or via `--set data.loader.batch_size=8`.
- **"No images found" Errors**: Verify folder paths in your config. Use `python scripts/check_data.py <folder_path>` to confirm PNG/JPG image presence.
- **Windows DataLoader Hangs**: Set `data.loader.num_workers: 0` in your YAML or via command line `--set data.loader.num_workers=0`.
- **Metrics Not Matching Published Papers**: This pipeline quantizes SR outputs to 8-bit, converts to ITU-R BT.601 Y-channel, and crops `scale` border pixels to match standard super-resolution benchmark evaluation standards.
- **Resuming After a Crash**: Training automatically writes progress to `train_log.csv` and saves `checkpoints/last.pth` every epoch. If a run crashes, rerun the script; use `--force` if `skip_if_done` is enabled.
