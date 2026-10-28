# Team guide: training on Kaggle and sharing results

One place for everything. Commands are for **Windows PowerShell** or **bash** run from the project root
(the folder that contains `src/`, `configs/`, `scripts/`, `kaggle/`). On Linux/macOS use `cp -r` instead of
`Copy-Item` and `~/.kaggle/` instead of `C:\Users\<you>\.kaggle\`.

Owner = Suresh (`suresh79061`). Friends = everyone else.

---

## Rules (read first)

1. **Use `.env` for your personal Kaggle setup.** Never commit personal API tokens, usernames, or runner scripts.
2. **Never edit `src/`, core `scripts/` or `configs/`** without discussing with the owner.
3. **Everyone uses the same shared settings** (below), otherwise models are not comparable.
4. **Commit only your own `results/<experiment_name>/` folders.** Never commit `summary.csv`, `summary.md`,
   `kaggle_out/` or anything from other people's folders.

### Shared settings (use exactly this unless the owner announces a change)

```python
EXTRA_SET = ["data.loader.num_workers=4", "data.train.repeat=3", "evaluation.save_samples=1"]
BRANCH = "exp-v6"      # the git tag/branch every run must use, so all models are trained with identical code
```

---

## A. Owner: one-time setup (do this before the friends start)

1. **Tag releases/experiments** so every remote run clones the exact same code:
   ```bash
   git tag exp-v6
   git push origin exp-v6
   ```
   The repo must be **public** (the Kaggle runner clones it without logging in).
2. **Share the Kaggle dataset** `suresh79061/sr-data` with each friend: open the dataset page, open its
   sharing / collaborator settings and add their Kaggle usernames as viewers. Keep it private.

---

## B. Friends: one-time setup

1. **Kaggle account.** Sign up at kaggle.com.
2. **Check you can open the dataset** `https://www.kaggle.com/datasets/suresh79061/sr-data` once the owner
   has shared it. If it says not found / no access, ask the owner to re-share.
3. **API token.** Kaggle website -> your profile -> Settings -> API -> **Create New Token**. This downloads
   `kaggle.json`. Move it to `~/.kaggle/kaggle.json` (or `C:\Users\<you>\.kaggle\kaggle.json` on Windows).
4. **Install the Kaggle CLI and test it:**
   ```bash
   pip install kaggle
   kaggle datasets files suresh79061/sr-data
   ```
   You should see the data folders listed.
5. **Get the repo:**
   ```bash
   git clone https://github.com/sureshkumarhere/Image-processing-Interpolation-using-neural-networks.git
   cd Image-processing-Interpolation-using-neural-networks
   ```
   You do **not** need the image datasets on your own computer. Kaggle already has them.
6. **Set up your local `.env`** (ignored by git, safe for credentials):
   Copy `.env.example` to `.env` and fill in your details:
   ```env
   KAGGLE_USERNAME=YOUR_KAGGLE_USERNAME
   KAGGLE_KEY=YOUR_KAGGLE_KEY
   KAGGLE_BRANCH=exp-v6
   KAGGLE_CONFIGS=configs/our_nn_x4_noise15_a1_skip.yaml
   ```
7. **Verify local setup:**
   ```bash
   python scripts/push_kaggle.py --prepare-only
   ```
   This generates `kaggle/kernel-metadata.json` locally without committing any secrets.

---

## C. Everyone: running a job on Kaggle

### Step 1: Push the kernel to Kaggle
```bash
python scripts/push_kaggle.py
```
*(Or specify configs on the fly: `python scripts/push_kaggle.py --configs configs/our_nn_x4_clean.yaml`)*

### Step 2: Check status
```bash
python scripts/push_kaggle.py --status
```
*(Or run `kaggle kernels status YOUR_KAGGLE_USERNAME/sr-run` or view in browser at `https://www.kaggle.com/code/YOUR_KAGGLE_USERNAME/sr-run`)*

### Step 3: Download the results
When the kernel finishes:
```bash
python scripts/push_kaggle.py --output
```
Outputs are downloaded to `kaggle_out/results/<experiment_name>/`.

---

## D. Everyone: sending results back to GitHub

Copy **only your experiment folders** into your clone's `results/` folder, then commit exactly those:

```powershell
# Windows PowerShell
Copy-Item kaggle_out\results\* results\ -Recurse -Force
git pull --rebase
git add results/our_nn_x4_noise15_a1_skip
git commit -m "Add results: our_nn_x4_noise15_a1_skip (friend1)"
git push
```

```bash
# Linux / macOS
cp -r kaggle_out/results/* results/
git pull --rebase
git add results/our_nn_x4_noise15_a1_skip
git commit -m "Add results: our_nn_x4_noise15_a1_skip (friend1)"
git push
```

*(Never commit `summary.csv`, `summary.md`, or `kaggle_out/`.)*

---

## E. Owner: collecting everything

```bash
git pull
python scripts/aggregate_results.py
```
Then open `notebooks/results_analysis.ipynb`. `results/summary.csv` and `summary.md` are generated on the
owner's computer only.
