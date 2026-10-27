# How to Run and Extend the Super-Resolution Benchmark

Welcome to the Image Super-Resolution (Interpolation) Benchmark project!

---

## 1. What This Project Is

This repository provides a modular, reproducible pipeline to compare image upscaling (super-resolution) methods.
It degrades clean High-Resolution (HR) ground-truth images into Low-Resolution (LR) inputs, upscales them using classical interpolation (nearest, bilinear, bicubic, lanczos) or neural networks (SRCNN, OurNN), and evaluates quality using quantitative metrics (PSNR, SSIM, LPIPS).

It also includes a polished, interactive web interface for live demonstrations and visual model comparisons.

### Directory Structure Overview
- `app/`: FastAPI backend and responsive HTML/CSS/JS frontend GUI.
- `configs/`: YAML experiment configs inheriting from shared `_base_*.yaml` files.
- `src/`: Core Python package divided into `data`, `models`, `losses`, `metrics`, `training`, `evaluation`, and `utils`.
- `scripts/`: Executable CLI tools (`run_app.py`, `train.py`, `evaluate.py`, `run_batch.py`, `aggregate_results.py`, `make_dummy_data.py`, `check_data.py`).
- `tests/`: Automated unit tests executable with `pytest`.
- `notebooks/`: Jupyter notebooks for visual and statistical analysis of results (`results_analysis.ipynb`).
- `data/` & `results/`: Input datasets and output logs/metrics/samples (git-managed with `.gitkeep`).

---

## 2. Web Interface (GUI) Guide

### Launch Command & Local URL
```bash
python scripts/run_app.py --host 0.0.0.0 --port 8000
```
Open your browser at **http://127.0.0.1:8000**. ( for normal internet )
for college etherent **http://{ your ip address }:8000**.

### Model Discovery Rules
The web application dynamically scans `results/` at runtime for eligible neural models. A model run is eligible only when:
1. It contains `checkpoints/best.pth`.
2. It contains `config_resolved.yaml` describing a single neural model.
3. Its model architecture can be instantiated by the project's model registry.

*Note:* Classical baselines (no neural weights) and smoke-test runs (`smoke_test*`) are automatically ignored. Newly added results matching these criteria appear automatically upon page refresh.

### Demo Workflow
1. **Upload**: Drag-and-drop or browse an image (PNG, JPEG, WebP, BMP).
2. **Select Mode**:
   - *LR Input Mode*: Treats upload as low-resolution input. Option to attach a separate ground-truth HR reference image.
   - *HR Reference Mode*: Treats upload as clean ground-truth reference, degrades it (bicubic downscale with optional Gaussian noise), and processes the generated LR input.
3. **Select Models & Device**: Choose discovered model checkpoints and hardware device (`auto`, `cpu`, or `cuda`).
4. **Run Benchmark**: View real-time progress and per-model inference timing.
5. **Interactive Comparison**:
   - Toggle between **Grid Cards**, **Before/After Slider**, and **Difference Map** views.
   - Toggle **100% Pixel View** for inspecting fine detail.
   - View sortable metrics table with directional badges and PSNR bar chart.
6. **Export Results**: Download individual SR PNGs or a single ZIP bundle containing all outputs, difference maps, `summary.json`, and `summary.csv`.

---

## 3. First-Time Setup

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

## 4. Smoke Test (Do This First, Takes <1 Minute)

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

---

## 5. Getting the Real Datasets

To run full benchmark comparisons, manually place standard HR image datasets under the `data/` directory:

Expected folder structure:
- `data/DIV2K_train_HR/`: DIV2K training set (800 HR images).
- `data/Set5/`: Set5 test set (5 HR images).
- `data/Set14/`: Set14 test/val set (14 HR images).
- `data/BSD100/`: BSD100 test set (100 HR images).
- `data/Urban100/`: Urban100 test set (100 HR images).

Verify folder contents and image dimensions using `check_data.py`:
```bash
python scripts/check_data.py data/Set5 data/Set14 data/DIV2K_train_HR
```

---

## 6. Running Real Experiments

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

---

## 7. Troubleshooting

- **CUDA Out of Memory**: Lower `data.loader.batch_size` or `data.train.patch_size` in the YAML config or via `--set data.loader.batch_size=8`.
- **"No eligible models found" in Web UI**: Run a neural training experiment (e.g. `configs/smoke_test_our_nn.yaml` or `configs/srcnn_x4_clean.yaml`) so that `results/<experiment>/checkpoints/best.pth` and `config_resolved.yaml` exist.
- **Metrics Not Matching Published Papers**: This pipeline quantizes SR outputs to 8-bit, converts to ITU-R BT.601 Y-channel, and crops `scale` border pixels to match standard super-resolution benchmark evaluation standards.
- **Optional LPIPS Metric**: If `lpips` is not installed, PSNR and SSIM will calculate normally and LPIPS will be labeled as unavailable in the UI. Install via `pip install lpips`.
