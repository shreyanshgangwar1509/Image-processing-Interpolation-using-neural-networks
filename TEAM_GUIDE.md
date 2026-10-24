# Team guide: training on Kaggle and sharing results

One place for everything. Commands are for **Windows PowerShell** run from the project root
(the folder that contains `src/`, `configs/`, `scripts/`, `kaggle/`). On Linux/macOS use `cp -r` instead of
`Copy-Item` and `~/.kaggle/` instead of `C:\Users\<you>\.kaggle\`.

Owner = Suresh (`suresh79061`). Friends = everyone else.

---

## Rules (read first)

1. **Only edit your own folder** `kaggle/<yourname>/`. Never edit `src/`, `scripts/` or `configs/`
   (if you think something there is wrong, tell the owner).
2. **Everyone uses the same shared settings** (below), otherwise models are not comparable.
3. **Commit only your own `results/<experiment_name>/` folders.** Never commit `summary.csv`, `summary.md`,
   `kaggle_out/` or anything from other people's folders.


### Shared settings (use exactly this unless the owner announces a change)

```python
EXTRA_SET = ["data.loader.num_workers=4", "data.train.repeat=3", "evaluation.save_samples=1"]
BRANCH = "exp-v1"      # the git tag every run must use, so all models are trained with identical code
```

## A. Owner: one-time setup (do this before the friends start)

1. **Per-person folders** (so nobody edits the same file):
   ```
   mkdir kaggle\suresh
   git mv kaggle\run_on_kaggle.py kaggle\suresh\run_on_kaggle.py
   git mv kaggle\kernel-metadata.json kaggle\suresh\kernel-metadata.json
   ```
   Make sure `kaggle\suresh\run_on_kaggle.py` is the **latest runner** (it copies only new results and
   syncs every 5 minutes).
2. **Ignore files nobody should commit.** Add these lines to `.gitignore`:
   ```
   kaggle_out/
   results/smoke_test*/
   ```
   and untrack the old smoke test: `git rm -r --cached results/smoke_test`
3. **Commit, push, then tag** so every run uses the same code:
   ```
   git add -A
   git commit -m "Kaggle team setup"
   git push
   git tag exp-v1
   git push origin exp-v1
   ```
   The repo must be **public** (the Kaggle runner clones it without logging in).
4. **Share the Kaggle dataset** `suresh79061/sr-data` with each friend: open the dataset page, open its
   sharing / collaborator settings and add their Kaggle usernames as viewers (the wording on the page may
   differ). Keep it private, do not make it public.

---

## B. Friends: one-time setup

1. **Kaggle account.** Sign up at kaggle.com.
2. **Check you can open the dataset** `https://www.kaggle.com/datasets/suresh79061/sr-data` once the owner
   has shared it. If it says not found / no access, ask the owner to re-share.
3. **API token.** Kaggle website -> your profile -> Settings -> API -> **Create New Token**. This downloads
   `kaggle.json`. Move it to `C:\Users\<you>\.kaggle\kaggle.json` (create the `.kaggle` folder if needed).
   If the site gives you a different kind of token or tells you to set an environment variable instead,
   follow what it says.
4. **Install the CLI and test it:**
   ```
   pip install kaggle
   kaggle datasets files suresh79061/sr-data
   ```
   You should see the data folders listed. (If that command does not exist in your version, run
   `kaggle datasets list -s div2k` just to check the token works.)
5. **Get the repo** (you are a collaborator):
   ```
   git clone https://github.com/sureshkumarhere/Image-processing-Interpolation-using-neural-networks.git
   cd Image-processing-Interpolation-using-neural-networks
   ```
   You do **not** need the image datasets on your own computer. Kaggle already has them.
6. **Find your folder and copy files from my folder.**
7. **Edit `kaggle\friend1\kernel-metadata.json`:** change `id` to `YOUR_KAGGLE_USERNAME/sr-run`.
   Leave `dataset_sources` as `suresh79061/sr-data`.
8. **Edit `kaggle\friend1\run_on_kaggle.py`** (the block at the top). Edit each existing line in place;
   do not paste a second copy of a line, because the **last** assignment in the file wins:
   - `BRANCH = "exp-v1"`
   - `CONFIGS = [...]`: put the yaml name in config that you want to run.

9. **Commit your folder:**
   ```
   git add kaggle/friend1
   git commit -m "Add friend1 kaggle folder"
   git pull --rebase
   git push
   ```

---

## C. Everyone: running a job on Kaggle

### Step 1: 
```
kaggle kernels push -p kaggle/friend1
kaggle kernels status YOUR_KAGGLE_USERNAME/sr-run
```
Repeat the status command until it says complete (or error). Then:
```
kaggle kernels output YOUR_KAGGLE_USERNAME/sr-run -p kaggle_out
Get-Content kaggle_out\run_summary.txt

(or open `https://www.kaggle.com/code/YOUR_KAGGLE_USERNAME/sr-run` in the browser and read the log there).
```
### Step 2: download the results

```
kaggle kernels output YOUR_KAGGLE_USERNAME/sr-run -p kaggle_out
```
You will get `kaggle_out\results\<experiment_name>\` for each config that ran, plus `run_summary.txt` and
`sr-run.log`. Each experiment folder contains `metrics.csv`, `summary.json`, `train_log.csv`, `log.txt`,
`config_resolved.yaml`, `env.txt`, `samples/` and `checkpoints/`.

---

## D. Everyone: sending results back

### 1. Results (small files) go to GitHub

Copy **only your experiment folders** into your clone's `results/` folder, then commit exactly those:
```
Copy-Item kaggle_out\results\* results\ -Recurse -Force
git pull --rebase
git add results/our_nn_x4_noise15 results/our_nn_x2_clean
git commit -m "Add results: our_nn_x4_noise15, our_nn_x2_clean (friend1)"
git push
```
(Use your own experiment folder names in the `git add` line. Do not use `git add -A` or `git add .`.)
Because everyone only adds their own folders, there are no merge conflicts. If `git push` is rejected,
run `git pull --rebase` and push again.

## E. Owner: collecting everything

```
git pull
python scripts/aggregate_results.py
```
Then open `notebooks/results_analysis.ipynb`. `results/summary.csv` and `summary.md` are generated on the
owner's computer only.

