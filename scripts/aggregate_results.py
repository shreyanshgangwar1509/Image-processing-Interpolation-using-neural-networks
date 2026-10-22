"""Scans results/ subfolders, aggregates experiment summaries into CSV and Markdown reports."""
import json
from pathlib import Path
import sys
import pandas as pd
import yaml

# Insert repo root into sys.path
repo_root = Path(__file__).parent.parent.resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def extract_noise_sigma(degradation_list: list) -> float:
    """Extracts noise sigma from degradation pipeline steps config if present, else 0.0."""
    if not degradation_list:
        return 0.0
    for step in degradation_list:
        if isinstance(step, dict) and step.get("name") == "gaussian_noise":
            return float(step.get("sigma", 0.0))
    return 0.0


def aggregate_results() -> None:
    results_dir = Path("results")
    if not results_dir.exists():
        print("No results/ directory found.")
        return

    summary_rows = []

    for exp_dir in results_dir.iterdir():
        if not exp_dir.is_dir():
            continue

        summary_json_path = exp_dir / "summary.json"
        config_yaml_path = exp_dir / "config_resolved.yaml"

        if not summary_json_path.exists():
            continue

        try:
            with open(summary_json_path, "r", encoding="utf-8") as f:
                sum_data = json.load(f)

            noise_sigma = 0.0
            if config_yaml_path.exists():
                with open(config_yaml_path, "r", encoding="utf-8") as f:
                    cfg_data = yaml.safe_load(f) or {}
                noise_sigma = extract_noise_sigma(cfg_data.get("degradation", []))

            exp_name = sum_data.get("experiment_name", exp_dir.name)
            scale = sum_data.get("scale", None)
            degradation_tag = sum_data.get("degradation", "")

            for item in sum_data.get("summary", []):
                model_label = item.get("model", "unknown")
                dataset_name = item.get("dataset", "unknown")
                n_images = item.get("n_images", 0)
                param_count = item.get("param_count", 0)
                mean_inf_time = item.get("mean_inference_time_s", 0.0)

                metrics = item.get("metrics", {})

                row = {
                    "experiment": exp_name,
                    "model": model_label,
                    "dataset": dataset_name,
                    "scale": scale,
                    "degradation": degradation_tag,
                    "noise_sigma": noise_sigma,
                    "n_images": n_images,
                    "param_count": param_count,
                    "mean_inference_time_s": mean_inf_time,
                }

                for metric_name, metric_val in metrics.items():
                    row[f"mean_{metric_name}"] = metric_val

                summary_rows.append(row)

        except Exception as e:
            print(f"[WARNING] Error reading results from '{exp_dir}': {e}")

    if not summary_rows:
        print("No summary.json files found in results/ subdirectories.")
        return

    df_summary = pd.DataFrame(summary_rows)

    # Reorder columns
    base_cols = [
        "experiment",
        "model",
        "dataset",
        "scale",
        "degradation",
        "noise_sigma",
        "n_images",
        "param_count",
        "mean_inference_time_s",
    ]
    metric_cols = [c for c in df_summary.columns if c not in base_cols]
    ordered_cols = base_cols + sorted(metric_cols)
    df_summary = df_summary[ordered_cols]

    # Save results/summary.csv
    csv_out = results_dir / "summary.csv"
    df_summary.to_csv(csv_out, index=False)
    print(f"Aggregated summary CSV saved to: {csv_out}")

    # Generate results/summary.md
    md_lines = ["# Super-Resolution Benchmark Aggregated Results\n"]

    unique_datasets = sorted(df_summary["dataset"].unique())
    for ds in unique_datasets:
        md_lines.append(f"## Dataset: {ds}\n")
        ds_df = df_summary[df_summary["dataset"] == ds].copy()

        # Sort by PSNR descending if mean_psnr exists
        if "mean_psnr" in ds_df.columns:
            ds_df.sort_values(by="mean_psnr", ascending=False, inplace=True)

        md_lines.append(ds_df.to_markdown(index=False))
        md_lines.append("\n")

    md_out = results_dir / "summary.md"
    with open(md_out, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"Aggregated summary Markdown saved to: {md_out}")


def main() -> None:
    aggregate_results()


if __name__ == "__main__":
    main()
