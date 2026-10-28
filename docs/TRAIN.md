# Training & Experimentation Guide

This guide details local model training, batch execution, remote cloud execution on Kaggle GPUs, and results aggregation.

---

## 1. Local Training & Evaluation

All experiments are driven by modular YAML configs under `configs/` that inherit from base configurations (`_base_x2.yaml`, `_base_x4.yaml`).

### Training a Single Model

```bash
# Train SRCNN x4 on clean data
python scripts/train.py --config configs/srcnn_x4_clean.yaml

# Train OurNN x4 on noisy data (sigma=15)
python scripts/train.py --config configs/our_nn_x4_noise15.yaml
```

### Overriding Config Parameters from the CLI

Any YAML key can be overridden dynamically using `--set key=value`:

```bash
python scripts/train.py --config configs/srcnn_x4_clean.yaml \
  --set training.epochs=5 \
  --set data.loader.batch_size=8 \
  --set training.lr=0.0001
```

### Running Batch Experiments

Run a series of experiments sequentially:

```bash
# Run all OurNN clean ablations
python scripts/run_batch.py configs/our_nn_x4_clean_a*.yaml

# Run all SRCNN baselines
python scripts/run_batch.py configs/srcnn_*.yaml
```

### Evaluating Existing Checkpoints

To evaluate a trained checkpoint across benchmark datasets without retraining:

```bash
python scripts/evaluate.py \
  --config results/our_nn_x4_clean/config_resolved.yaml \
  --checkpoint results/our_nn_x4_clean/checkpoints/best.pth
```

---

## 2. Remote Cloud Training on Kaggle

When local GPU acceleration is not available, training jobs can be offloaded to free Kaggle GPU instances using the project's automated Kaggle orchestration runner.

### Shared Benchmark Protocol
To ensure all benchmark models remain directly comparable, every run must adhere to these settings:

```python
EXTRA_SET = ["data.loader.num_workers=4", "data.train.repeat=3", "evaluation.save_samples=1"]
BRANCH = "exp-v6"  # The git tag/branch cloned on Kaggle so all models use identical code
```

### Step 1: One-Time Authentication
1. Log in to [Kaggle](https://kaggle.com) -> **Settings** -> **API** -> **Create New Token**. This downloads `kaggle.json`.
2. Place `kaggle.json` in your local Kaggle directory:
   * **Linux / macOS**: `~/.kaggle/kaggle.json`
   * **Windows**: `C:\Users\<YourUsername>\.kaggle\kaggle.json`
3. *(Alternative)* Or configure your `.env` file in the project root:
   ```env
   KAGGLE_USERNAME=your_kaggle_username
   KAGGLE_KEY=your_kaggle_key
   KAGGLE_BRANCH=exp-v6
   KAGGLE_CONFIGS=configs/our_nn_x4_noise15_a1_skip.yaml
   ```

### Step 2: Push Training Job to Kaggle

Use the cross-platform CLI helper [`scripts/push_kaggle.py`](file:///home/ultra/code/github/Image-processing-Interpolation-using-neural-networks/scripts/push_kaggle.py):

```bash
# Push default config specified in .env or script
python scripts/push_kaggle.py

# Or specify configs on the fly
python scripts/push_kaggle.py --configs configs/our_nn_x4_clean_a1_skip.yaml
```

The script automatically detects your Kaggle credentials, generates `kaggle/kernel-metadata.json`, updates the runner, and submits the job.

### Step 3: Monitor & Download Outputs

```bash
# Check remote status
python scripts/push_kaggle.py --status

# Download outputs once complete
python scripts/push_kaggle.py --output
```

Results are downloaded to `kaggle_out/results/<experiment_name>/`, including:
* `checkpoints/best.pth` and `checkpoints/last.pth`
* `metrics.csv` (per-dataset PSNR, SSIM, and timings)
* `train_log.csv` (per-epoch loss and validation metrics)
* `config_resolved.yaml` (full snapshot of runtime settings)
* Visual prediction samples under `samples/`

---

## 3. Results Sharing & Git Collaboration

To prevent merge conflicts and avoid repository bloat:

1. **Commit only your experiment folders**:
   ```bash
   # Copy results from kaggle_out/ to results/
   cp -r kaggle_out/results/our_nn_x4_clean_a1_skip results/

   # Commit strictly that experiment directory
   git pull --rebase
   git add results/our_nn_x4_clean_a1_skip
   git commit -m "Add results: our_nn_x4_clean_a1_skip"
   git push
   ```
2. **Never commit transient files**:
   * Do not commit `kaggle_out/`, `*.log`, `run_summary.txt`, or smoke test outputs.
   * `summary.csv` and `summary.md` are compiled locally by the repository maintainer.

---

## 4. Aggregating & Visualizing Results

Once experiment folders are placed under `results/`:

### Compile Benchmark Tables

```bash
python scripts/aggregate_results.py
```
This generates:
* `results/summary.csv`: Comprehensive tabular metrics across all models, datasets, noise levels, parameter counts, and inference speeds.
* `results/summary.md`: Clean Markdown comparison table.

### Interactive Analysis Notebook

Open `notebooks/results_analysis.ipynb` in Jupyter:
* Generates bar charts comparing classical baselines vs. deep models.
* Generates parameter count vs. PSNR efficiency curves.
* Produces ablation progression figures (a0 -> a1 -> a2 -> full).
* Exports publication-ready figures to `notebooks/report_figures/`.
