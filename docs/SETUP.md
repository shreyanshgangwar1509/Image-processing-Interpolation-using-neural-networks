# Setup & Installation Guide

This guide covers environment installation, dataset acquisition, sanity checks, and launching the web interface.

---

## 1. Environment Installation

The project supports Python 3.10 through 3.13 on Linux, macOS, and Windows.

### Linux / macOS

```bash
# 1. Clone repository
git clone https://github.com/sureshkumarhere/Image-processing-Interpolation-using-neural-networks.git
cd Image-processing-Interpolation-using-neural-networks

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Windows (PowerShell)

```powershell
# 1. Clone repository
git clone https://github.com/sureshkumarhere/Image-processing-Interpolation-using-neural-networks.git
cd Image-processing-Interpolation-using-neural-networks

# 2. Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt
```

### GPU / CUDA Acceleration (Optional)

To enable CUDA acceleration for GPU training or faster inference, install the matching PyTorch CUDA wheel before installing requirements:

```bash
# Example for CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Verify GPU availability:
```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## 2. Dataset Acquisition

The benchmark evaluates models on clean High-Resolution (HR) ground-truth images degraded at runtime.

### Download Links

| Dataset | Split | Images | Download Link | Target Directory |
|---|---|---|---|---|
| **DIV2K_train_HR** | Train | 800 | [Google Drive (3.3 GB)](https://drive.google.com/file/d/172UOwD8apo8iCaBMXmr3vjeYqGn2ZIBo/view?usp=sharing) | `data/DIV2K_train_HR/` |
| **Set5** | Test | 5 | [Google Drive (ZIP)](https://drive.google.com/file/d/1Uf9X0AT7WozUBIQMhgyn9tcApTVcwyWE/view?usp=sharing) | `data/Set5/` |
| **Set14** | Test / Val | 14 | Included in Benchmark ZIP above | `data/Set14/` |
| **BSD100** | Test | 100 | Included in Benchmark ZIP above | `data/BSD100/` |
| **Urban100** | Test | 100 | Included in Benchmark ZIP above | `data/Urban100/` |

*(Note: If you are training via Kaggle, datasets are already pre-hosted on Kaggle at `suresh79061/sr-data`.)*

### Expected Folder Structure

```text
data/
├── DIV2K_train_HR/      # 0001.png ... 0800.png
├── Set5/                # baby.png, bird.png, butterfly.png, head.png, woman.png
├── Set14/               # baboon.png, barbara.png, bridge.png ...
├── BSD100/              # 101085.png, 101087.png ...
└── Urban100/            # img_001.png ... img_100.png
```

Verify your dataset paths and image dimensions:
```bash
python scripts/check_data.py data/Set5 data/Set14 data/DIV2K_train_HR
```

---

## 3. Quick Verification (Smoke Test)

Run these quick checks (< 1 minute on CPU) to verify that data loaders, neural models, and metric pipelines are functioning:

```bash
# 1. Generate synthetic dummy images for testing
python scripts/make_dummy_data.py --out data/dummy

# 2. Run CPU smoke training test
python scripts/train.py --config configs/smoke_test.yaml

# 3. Run unit tests
pytest
```

---

## 4. Interactive Web Interface (GUI)

The repository includes a modern web interface for interactive visual comparison and metric evaluation.

### Launch Command

```bash
python scripts/run_app.py --host 0.0.0.0 --port 8000
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser (or `http://<your-ip>:8000` for local network access).

### Features
* **LR Input Mode**: Upload a low-resolution image to upscale with classical or neural models. Optionally attach a reference ground truth image for metrics.
* **HR Reference Mode**: Upload a high-resolution image; the pipeline degrades it (bicubic downscaling with optional Gaussian noise) and benchmarks the upscaled outputs against the original.
* **Interactive Views**:
  * **Card Grid**: Side-by-side display of all model outputs with timing and metrics.
  * **Split-Slider**: Compare two models with an interactive slider.
  * **Difference Map**: Visual heatmap highlighting error residuals.
  * **100% Pixel Crop**: Zoom in on fine edges and textures.
* **Export**: Download individual PNG outputs or a ZIP archive containing all images, difference maps, `summary.json`, and `summary.csv`.

### Dynamic Model Discovery
The web app automatically searches `results/`, `weights/`, and `checkpoints/` for trained models. Any directory containing:
1. Model weights (`checkpoints/best.pth` or `*.pth`)
2. Configuration file (`config_resolved.yaml` or `config.yaml`)

will automatically appear in the model selector upon page refresh.

---

## 5. Troubleshooting

* **CUDA Out of Memory**: Reduce batch size or patch size in your YAML config or via command-line override:
  ```bash
  python scripts/train.py --config configs/our_nn_x4_clean.yaml --set data.loader.batch_size=8 --set data.train.patch_size=48
  ```
* **Read-only File System in Sandbox**: Ensure write operations run in an environment with appropriate filesystem write permissions.
* **LPIPS Metric Unavailable**: LPIPS is an optional perceptual metric. If not installed, PSNR and SSIM continue to calculate normally. Install via `pip install lpips`.
