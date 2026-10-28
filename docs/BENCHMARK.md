# Quantitative & Qualitative Benchmark Results

This document presents the complete evaluation results for the Image Super-Resolution Benchmark, comparing classical interpolation baselines, reference neural networks (SRCNN), and the custom `OurNN` model across standard benchmark datasets under clean and noisy degradation conditions.

---

## 1. Experimental Setup & Protocol

All evaluations follow standard super-resolution evaluation conventions established in literature:

1. **Color Space**: Both Super-Resolved (SR) predictions and High-Resolution (HR) ground-truth images are converted to the Y-luminance channel using the ITU-R BT.601 standard.
2. **Quantization**: Values are clamped to the valid range `[0, 255]` and rounded to 8-bit integers before metric evaluation.
3. **Border Cropping**: Image boundaries are cropped by `scale` pixels on all four sides to eliminate border padding effects.
4. **Metrics**:
   * **PSNR (Peak Signal-to-Noise Ratio)**: Evaluated in decibels (dB). Higher values indicate lower pixel-level reconstruction error.
   * **SSIM (Structural Similarity Index)**: Bounded between 0.0 and 1.0. Higher values indicate better structural, contrast, and luminance fidelity.
   * **Inference Runtime**: Measured in milliseconds (ms) averaged over all images in the respective test dataset.

### Evaluated Benchmark Datasets

* **Set5** (5 images): Standard small test set featuring portraits, animals, and objects with distinct high-frequency textures.
* **Set14** (14 images): Diverse test set containing complex patterns, natural landscapes, and human subjects.
* **BSD100** (100 images): Natural images from the Berkeley Segmentation Dataset featuring rich natural backgrounds, foliage, and textures.
* **Urban100** (100 images): Challenging urban and architectural dataset rich in repetitive geometric structures, grid patterns, and sharp high-contrast edges.

---

## 2. Quantitative Results

### Scale x2 Clean Benchmark (Bicubic Downsampling)

Under x2 clean downsampling, `OurNN` outperforms all classical baselines and SRCNN by a substantial margin across all benchmark datasets.

| Method | Parameters | Set5 (PSNR / SSIM) | Set14 (PSNR / SSIM) | BSD100 (PSNR / SSIM) | Urban100 (PSNR / SSIM) | Runtime (ms) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **OurNN (x2 Clean)** | **165.3k** | **37.10 dB / 0.9572** | **32.69 dB / 0.9097** | **31.49 dB / 0.8910** | **29.68 dB / 0.8997** | 19.1 ms |
| **Lanczos** | - | 34.32 dB / 0.9365 | 30.78 dB / 0.8800 | 29.92 dB / 0.8550 | 27.26 dB / 0.8505 | 11.2 ms |
| **SRCNN (x2 Clean)** | 20.1k | 34.15 dB / 0.9318 | 30.70 dB / 0.8771 | 29.89 dB / 0.8538 | 27.24 dB / 0.8470 | 13.9 ms |
| **Bicubic** | - | 33.98 dB / 0.9332 | 30.56 dB / 0.8752 | 29.75 dB / 0.8497 | 27.07 dB / 0.8459 | 3.5 ms |
| **Bilinear** | - | 32.25 dB / 0.9119 | 29.25 dB / 0.8407 | 28.69 dB / 0.8106 | 25.97 dB / 0.8090 | 1.1 ms |
| **Nearest Neighbor** | - | 30.86 dB / 0.9002 | 28.58 dB / 0.8464 | 28.42 dB / 0.8253 | 25.63 dB / 0.8211 | 0.6 ms |

* **Key Takeaway**: On Set5, `OurNN` achieves **37.10 dB**, outperforming SRCNN (+2.95 dB) and Bicubic (+3.12 dB). On Urban100, `OurNN` achieves **29.68 dB** (+2.44 dB over SRCNN), demonstrating superior reconstruction of sharp architectural lines.

---

### Scale x4 Clean Benchmark (Bicubic Downsampling)

At the challenging x4 scale factor, 75% of the original spatial pixels are discarded. `OurNN` and its architectural variants show consistent improvements over SRCNN and classical baselines.

| Method | Parameters | Set5 (PSNR / SSIM) | Set14 (PSNR / SSIM) | BSD100 (PSNR / SSIM) | Urban100 (PSNR / SSIM) | Runtime (ms) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **OurNN (Full Model)** | **202.3k** | **30.56 dB / 0.8655** | **27.56 dB / 0.7559** | **26.90 dB / 0.7137** | **24.41 dB / 0.7211** | 12.6 ms |
| **OurNN (A1: Skip Only)** | 196.0k | 30.42 dB / 0.8621 | 27.46 dB / 0.7532 | 26.86 dB / 0.7119 | 24.31 dB / 0.7164 | 10.6 ms |
| **OurNN (A2: Attention Only)** | 199.3k | 30.40 dB / 0.8616 | 27.45 dB / 0.7528 | 26.85 dB / 0.7112 | 24.30 dB / 0.7160 | 11.3 ms |
| **OurNN (A0: Plain CNN)** | 196.0k | 29.50 dB / 0.8353 | 26.82 dB / 0.7350 | 26.50 dB / 0.6987 | 23.73 dB / 0.6883 | 10.0 ms |
| **SRCNN (x4 Clean)** | 20.1k | 29.19 dB / 0.8291 | 26.60 dB / 0.7244 | 26.34 dB / 0.6877 | 23.58 dB / 0.6805 | 14.2 ms |
| **Lanczos** | - | 28.81 dB / 0.8182 | 26.32 dB / 0.7130 | 26.13 dB / 0.6758 | 23.33 dB / 0.6654 | 9.2 ms |
| **Bicubic** | - | 28.63 dB / 0.8138 | 26.22 dB / 0.7089 | 26.05 / 0.6722 | 23.24 dB / 0.6616 | 3.6 ms |
| **Bilinear** | - | 27.56 dB / 0.7896 | 25.52 dB / 0.6821 | 25.55 dB / 0.6469 | 22.69 dB / 0.6354 | 1.2 ms |
| **Nearest Neighbor** | - | 26.26 dB / 0.7380 | 24.74 dB / 0.6553 | 25.04 dB / 0.6300 | 22.17 dB / 0.6160 | 0.5 ms |

* **Key Takeaway**: The full `OurNN` model achieves **30.56 dB / 0.8655** on Set5, outperforming SRCNN by **+1.37 dB** and Bicubic by **+1.93 dB**. On Urban100, `OurNN` scores **24.41 dB**, providing a significant boost over SRCNN (23.58 dB) and Bicubic (23.24 dB).

![Results Overview](../notebooks/report_figures/fig2_results.png)

---

### Scale x4 Noisy Benchmark (Bicubic Downsampling + Gaussian Noise sigma=15)

In this regime, additive zero-mean Gaussian noise (`sigma = 15`) is injected into the low-resolution input, testing joint denoising and super-resolution robustness.

| Method | Parameters | Set5 (PSNR / SSIM) | Set14 (PSNR / SSIM) | BSD100 (PSNR / SSIM) | Urban100 (PSNR / SSIM) | Runtime (ms) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **OurNN (x4 Noise 15)** | **202.3k** | **27.89 dB / 0.7873** | **25.82 dB / 0.6779** | **25.55 dB / 0.6368** | **23.26 dB / 0.6537** | 12.8 ms |
| **OurNN (A1: Skip Only)** | 196.0k | 27.84 dB / 0.7854 | 25.77 dB / 0.6754 | 25.54 dB / 0.6355 | 23.21 dB / 0.6513 | 10.5 ms |
| **OurNN (A2: Attention Only)** | 199.3k | 27.75 dB / 0.7809 | 25.73 dB / 0.6750 | 25.52 dB / 0.6354 | 23.19 dB / 0.6499 | 11.4 ms |
| **OurNN (A0: Plain CNN)** | 196.0k | 27.50 dB / 0.7738 | 25.52 dB / 0.6684 | 25.41 dB / 0.6308 | 23.01 dB / 0.6415 | 9.9 ms |
| **SRCNN (x4 Noise 15)** | 20.1k | 27.39 dB / 0.7724 | 25.46 dB / 0.6613 | 25.30 dB / 0.6235 | 22.88 dB / 0.6326 | 14.1 ms |
| **Bicubic** | - | 26.38 dB / 0.6971 | 24.74 dB / 0.6047 | 24.59 dB / 0.5660 | 22.40 dB / 0.5639 | 3.6 ms |
| **Lanczos** | - | 26.35 dB / 0.6917 | 24.71 dB / 0.6002 | 24.54 dB / 0.5609 | 22.40 dB / 0.5600 | 9.4 ms |
| **Bilinear** | - | 26.28 dB / 0.7179 | 24.67 dB / 0.6189 | 24.68 dB / 0.5826 | 22.21 dB / 0.5749 | 1.2 ms |
| **Nearest Neighbor** | - | 24.38 dB / 0.5800 | 23.36 dB / 0.5138 | 23.52 dB / 0.4840 | 21.32 dB / 0.4812 | 0.5 ms |

* **Key Takeaway**: Classical interpolation methods amplify noise artifacts (Bicubic drops to 26.38 dB on Set5). `OurNN` learns to suppress high-frequency noise while recovering underlying structural edges, delivering **27.89 dB** (+1.51 dB over Bicubic and +0.50 dB over SRCNN).

---

## 3. Systematic Ablation Study

To analyze the contribution of each architectural innovation in `OurNN`, we conducted a controlled ablation study on the x4 clean super-resolution benchmark:

```mermaid
flowchart TD
    A0["A0: Plain CNN\n(No residual skip, no attention)\n29.50 dB (Set5)"] -->|"+ Global Bicubic Residual Skip"| A1["A1: Skip Only\n(+0.92 dB gain)\n30.42 dB (Set5)"]
    A0 -->|"+ Channel Attention (SE)"| A2["A2: Attention Only\n(+0.90 dB gain)\n30.40 dB (Set5)"]
    A1 -->|"+ Channel Attention"| Full["Full OurNN (A3)\n(Skip + Attention)\n30.56 dB (Set5)"]
    A2 -->|"+ Residual Skip"| Full
```

### Ablation Comparison Table (Set5, Scale x4 Clean)

| Variant | Residual Skip | Channel Attention | Edge Priors | Parameters | PSNR (dB) | SSIM | PSNR Gain |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **A0: Plain Baseline** | No | No | Fixed Sobel/Laplacian | 196.0k | 29.50 | 0.8353 | Baseline |
| **A1: Residual Skip** | **Yes** | No | Fixed Sobel/Laplacian | 196.0k | 30.42 | 0.8621 | **+0.92 dB** |
| **A2: Channel Attention** | No | **Yes** | Fixed Sobel/Laplacian | 199.3k | 30.40 | 0.8616 | **+0.90 dB** |
| **Full OurNN** | **Yes** | **Yes** | Fixed Sobel/Laplacian | **202.3k** | **30.56** | **0.8655** | **+1.06 dB** |

### Insights from Ablation Analysis

1. **Impact of Global Residual Skip (A1)**:
   Adding the global bicubic skip connection produced the single largest performance jump (**+0.92 dB** on Set5, **+0.64 dB** on Set14). Rather than forcing the neural network to reconstruct low-frequency color and coarse structure, the network only learns the high-frequency residual difference between the bicubic upscale and the ground truth. This stabilizes gradients and speeds up convergence.
2. **Impact of Channel Attention (A2)**:
   The Squeeze-and-Excitation (SE) block adaptively recalibrates feature maps, increasing weights on informative edge channels while suppressing uninformative activations. This provides a **+0.90 dB** boost over the plain CNN with only 3.3k additional parameters.
3. **Compound Benefit (Full Model)**:
   Combining both skip connections and channel attention delivers the highest fidelity (**30.56 dB** on Set5, **24.41 dB** on Urban100), confirming that the two mechanisms are complementary.

![Ablation Study Comparison](../notebooks/report_figures/fig3_ablation.png)

---

## 4. Visual Qualitative Comparisons

Quantitative metrics such as PSNR and SSIM measure average pixel fidelity, but qualitative visual inspection reveals how well fine textures, straight architectural edges, and text are preserved.

![Qualitative Visual Comparisons](../notebooks/report_figures/fig4_qualitative.png)

### Visual Findings

* **Bicubic Interpolation**: Suffers from heavy blurring along high-contrast boundaries, rounded corners on geometric shapes, and severe ringing artifacts.
* **SRCNN**: Successfully sharpens coarse boundaries but struggles with fine repetitive textures and parallel line structures (producing staircasing and chromatic smearing).
* **OurNN (A0: Plain)**: Sharper than SRCNN, but exhibits occasional contrast fading in complex textures.
* **OurNN (Full Model)**: Produces crisp, well-defined straight edges and restores repetitive grid textures without unnatural hallucination or over-smoothing.

---

## 5. Parameter Efficiency & Runtime Analysis

`OurNN` is designed with practical deployment in mind, achieving competitive super-resolution quality with a compact parameter footprint:

| Model | Parameters | Set5 x4 PSNR | Relative Size vs SRCNN | Average GPU Runtime |
|:---|:---:|:---:|:---:|:---:|
| **SRCNN** | 20.1k | 29.19 dB | 1.0x | ~14 ms |
| **OurNN (Full)** | **202.3k** | **30.56 dB** | **10.0x** | **~12 ms** |
| **Typical EDSR** | ~43,000k | ~32.4 dB | ~2100x | ~150 ms |

![Parameter Efficiency vs PSNR](../notebooks/report_figures/extra_params_vs_psnr.png)

* Despite having 10x more parameters than the minimal 3-layer SRCNN, `OurNN` operates at comparable or faster runtime (~12 ms per image) thanks to modern grouped/depth-wise tensor operations and efficient memory layout.
* Compared to heavy modern super-resolution architectures with tens of millions of parameters, `OurNN` delivers strong reconstruction quality within an embedded-friendly ~200k parameter budget.

---

## 6. How to Reproduce Benchmark Results

You can independently evaluate any checkpoint or reproduce the summary tables using the repository's evaluation tools:

### Evaluating a Checkpoint Across Test Sets

```bash
# Evaluate OurNN x4 Clean across all standard test sets
python scripts/evaluate.py \
  --config results/our_nn_x4_clean/config_resolved.yaml \
  --checkpoint results/our_nn_x4_clean/checkpoints/best.pth

# Evaluate OurNN x4 Noisy across all standard test sets
python scripts/evaluate.py \
  --config results/our_nn_x4_noise15/config_resolved.yaml \
  --checkpoint results/our_nn_x4_noise15/checkpoints/best.pth
```

### Re-aggregating All Benchmark Results

```bash
# Parse all experiment directories in results/ and update summary.csv / summary.md
python scripts/compile_results.py
```

### Interactive Visual Verification

Launch the web application to visually inspect outputs, compare models side-by-side, and inspect crops with interactive zoom and pan:

```bash
python scripts/run_app.py --host 127.0.0.1 --port 8000
```
Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.
