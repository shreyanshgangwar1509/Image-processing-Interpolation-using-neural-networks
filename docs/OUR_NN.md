# OurNN Model Architecture & Benchmark Guide

Welcome to the `OurNN` model documentation! `OurNN` is a lightweight, modular super-resolution neural network implemented in `src/models/our_nn.py`.

---

## 1. Model Overview & Architecture

`OurNN` combines residual block learning, fixed-kernel edge prior extraction, channel attention, sub-pixel upscaling, and global bicubic skip connections into a unified, lightweight architecture.

### ASCII Data-Flow Diagram

```
LR Input (B, 3, H, W)
  │
  ├─────────────────────────────────────────┐ (if bicubic_skip=True)
  │                                         │
  ▼                                         │
[Head Conv3x3] (B, n_feats, H, W)           │
  │                                         │
  ├───────── (if edge_branch=True) ──────┐  │
  │                                      │  │
  │     [Luminance Gray Map] (B, 1, H, W)│  │
  │             │                        │  │
  │     [Fixed Sobel-X, Y, Laplacian]    │  │
  │             │                        │  │
  │     [Concat Edge Maps] (B, 3, H, W)  │  │
  │             │                        │  │
  │     [Edge Conv3x3 + ReLU]            │  │
  │             │ (B, n_feats, H, W)     │  │
  │             ▼                        │  │
  └─────► [Concat & 1x1 Fuse Conv]       │  │
                │ (B, n_feats, H, W)     │  │
                ▼                        │  │
        [N Residual Blocks + SE Attn]    │  │
                │                        │  │
        [Body Conv3x3]                   │  │
                │ + Global Skip          │  │
                ▼                        │  │
        [Sub-Pixel Upsampler Conv]       │  │
                │ (B, n_feats, s*H, s*W) │  │
                ▼                        │  │
        [Tail Conv3x3]                   │  │
                │                        │
                ▼                        ▼
        Residual (B, 3, s*H, s*W)  +  Bicubic Upscale (B, 3, s*H, s*W)
                │                        │
                └───────────┬────────────┘
                            ▼
                  SR Output (B, 3, s*H, s*W)
```

### Model Parameters & Flags

| Flag / Parameter | Default | Description |
|---|---|---|
| `scale` | *Required* | Upscaling factor (supported: 2, 3, 4). |
| `n_feats` | `32` | Number of intermediate feature channels. |
| `n_blocks` | `6` | Number of residual blocks in the body. |
| `res_scale` | `0.1` | Scaling factor multiplied with residual branch outputs. |
| `reduction` | `4` | Reduction ratio for Channel Attention module. |
| `bicubic_skip` | `True` | Adds bicubic upscaled LR directly to tail output. |
| `attention` | `True` | Applies Squeeze-and-Excitation channel attention inside resblocks. |
| `edge_branch` | `True` | Extracts luminance and fixed Sobel/Laplacian features. |
| `zero_init_tail`| `True` | Initializes tail conv weights/bias to 0 if `bicubic_skip` is True. |

---

## 2. Related Ideas & Our Contribution

### Credits to Prior Literature
- **Residual Learning on Bicubic Upscale**: Inspired by VDSR (Kim et al.).
- **Residual Blocks without Batch Normalization**: Inspired by EDSR (Lim et al.).
- **Channel Attention (Squeeze-and-Excitation)**: Inspired by SE-Net (Hu et al.) / RCAN (Zhang et al.).
- **Sub-Pixel Convolution Upscaling**: Inspired by ESPCN (Shi et al.).

### Our Contribution
Our contribution lies in the specific modular combination of these techniques alongside a fixed edge-prior fusion branch that extracts structural guidance without adding trainable parameters.

---

## 3. Ablation Study Plan

We evaluate the incremental value of each architectural component across 4 systematic steps on both clean and noisy (sigma 15) datasets:

| Ablation Step | Config File | `bicubic_skip` | `attention` | `edge_branch` | Core Question Answered |
|---|---|---|---|---|---|
| **a0 (Plain)** | `our_nn_x4_clean_a0_plain.yaml` | `False` | `False` | `False` | Baseline performance of plain deep network. |
| **a1 (+Skip)** | `our_nn_x4_clean_a1_skip.yaml` | `True` | `False` | `False` | Benefit of learning residual on top of bicubic input. |
| **a2 (+Attn)** | `our_nn_x4_clean_a2_attn.yaml` | `True` | `True` | `False` | Impact of Squeeze-and-Excitation channel attention. |
| **Full Model** | `our_nn_x4_clean.yaml` | `True` | `True` | `True` | Value of fixed edge-prior feature fusion branch. |

---

## 4. Execution Commands

### Linux / macOS (Bash)

```bash
# 1. Quick CPU Smoke Test
python scripts/train.py --config configs/smoke_test_our_nn.yaml

# 2. SRCNN Baseline Runs
python scripts/run_batch.py configs/srcnn_x4_clean.yaml configs/srcnn_x2_clean.yaml configs/srcnn_x4_noise15.yaml

# 3. OurNN Main Benchmark Runs
python scripts/run_batch.py configs/our_nn_x4_clean.yaml configs/our_nn_x2_clean.yaml configs/our_nn_x4_noise15.yaml

# 4. OurNN Ablation Study Experiments
python scripts/run_batch.py configs/our_nn_x4_clean_a*.yaml configs/our_nn_x4_noise15_a*.yaml

# 5. Aggregate Results
python scripts/aggregate_results.py
```

### Windows (PowerShell)

```powershell
# 1. Quick CPU Smoke Test
python scripts/train.py --config configs/smoke_test_our_nn.yaml

# 2. SRCNN Baseline Runs
python scripts/run_batch.py configs/srcnn_x4_clean.yaml configs/srcnn_x2_clean.yaml configs/srcnn_x4_noise15.yaml

# 3. OurNN Main Benchmark Runs
python scripts/run_batch.py configs/our_nn_x4_clean.yaml configs/our_nn_x2_clean.yaml configs/our_nn_x4_noise15.yaml

# 4. OurNN Ablation Study Experiments
python scripts/run_batch.py configs/our_nn_x4_clean_a*.yaml configs/our_nn_x4_noise15_a*.yaml

# 5. Aggregate Results
python scripts/aggregate_results.py
```

---

## 5. Analyzing Results in `results/summary.csv`

The model name is `our_nn` for all variants, so the `experiment` column distinguishes ablation steps.

### Python / Pandas Snippet to Build Ablation Table

```python
import pandas as pd

df = pd.read_csv("results/summary.csv")

# Filter OurNN clean x4 ablations
ablation_exps = [
    "our_nn_x4_clean_a0_plain",
    "our_nn_x4_clean_a1_skip",
    "our_nn_x4_clean_a2_attn",
    "our_nn_x4_clean",
]

df_abl = df[df["experiment"].isin(ablation_exps)].copy()

# Pivot mean_psnr by experiment and dataset
piv = df_abl.pivot(index="experiment", columns="dataset", values="mean_psnr")
piv = piv.reindex(ablation_exps)

print("=== OurNN x4 Clean Ablation Study PSNR (dB) ===")
print(piv)
```

---

## 6. Suggested Run Order for Limited GPU Time

1. **Smoke test**: `python scripts/train.py --config configs/smoke_test_our_nn.yaml` (<1 min).
2. **Main OurNN x4 clean**: `python scripts/train.py --config configs/our_nn_x4_clean.yaml`.
3. **SRCNN x4 clean baseline**: `python scripts/train.py --config configs/srcnn_x4_clean.yaml`.
4. **Clean ablations**: `our_nn_x4_clean_a0_plain`, `a1_skip`, `a2_attn`.
5. **Noise15 experiments**: `our_nn_x4_noise15` and its ablations.
6. **Scale x2 experiments**: `our_nn_x2_clean.yaml`.

---

## 7. How to Add a New Idea Later

To extend `OurNN` without breaking backwards compatibility:
1. Open `src/models/our_nn.py` and add a new constructor flag (e.g., `use_new_feature: bool = False`).
2. Set the default value to `False` so all existing configs produce identical results.
3. Wrap your new module/logic inside `if self.use_new_feature:`.
4. Create a new YAML config file in `configs/` enabling the new parameter.
