"""Experiment orchestration runner."""
import json
import logging
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import yaml

from src.data import DATASETS
from src.data.degradation import DegradationPipeline
from src.evaluation.evaluator import count_parameters, evaluate_model
from src.models import MODELS
from src.training.trainer import Trainer
from src.utils.config import apply_overrides, load_config
from src.utils.device import get_device
from src.utils.logging_utils import setup_logging
from src.utils.seed import set_seed


def _get_env_info(device: torch.device) -> str:
    """Collects environment diagnostic information."""
    py_ver = sys.version.replace("\n", " ")
    torch_ver = torch.__version__
    dev_name = str(device)
    if device.type == "cuda":
        dev_name += f" ({torch.cuda.get_device_name(device)})"

    git_hash = "unknown"
    try:
        git_hash = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
    except Exception:
        pass

    env_str = (
        f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Platform: {platform.platform()}\n"
        f"Python: {py_ver}\n"
        f"PyTorch: {torch_ver}\n"
        f"Device: {dev_name}\n"
        f"Git Commit: {git_hash}\n"
    )
    return env_str


def run_experiment(
    cfg_or_path: Union[str, Path, Dict[str, Any]],
    overrides: Optional[List[str]] = None,
    overrides_already_applied: bool = False,
    checkpoint: Optional[Union[str, Path]] = None,
    eval_only: bool = False,
    device_override: Optional[str] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Single entry point for running super-resolution training and/or evaluation."""
    if isinstance(cfg_or_path, (str, Path)):
        cfg = load_config(cfg_or_path)
    else:
        cfg = dict(cfg_or_path)

    if overrides and not overrides_already_applied:
        cfg = apply_overrides(cfg, overrides)

    if device_override:
        cfg["device"] = device_override

    exp_name = cfg.get("experiment_name", "unnamed_experiment")
    output_dir = Path(cfg.get("output_dir", f"results/{exp_name}"))
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv = output_dir / "metrics.csv"
    summary_json = output_dir / "summary.json"

    skip_if_done = cfg.get("skip_if_done", False)
    if skip_if_done and not force and metrics_csv.exists() and summary_json.exists():
        print(f"Skipping experiment '{exp_name}' because results already exist in '{output_dir}'.")
        with open(summary_json, "r", encoding="utf-8") as f:
            return json.load(f)

    # Logging setup
    log_file = output_dir / "log.txt"
    logger = setup_logging(log_file)
    logger.info(f"--- Running Experiment: {exp_name} ---")

    # Seed and Device
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    device = get_device(cfg.get("device", "auto"))
    logger.info(f"Using device: {device}, seed: {seed}")

    # Write resolved config and env.txt
    with open(output_dir / "config_resolved.yaml", "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    with open(output_dir / "env.txt", "w", encoding="utf-8") as f:
        f.write(_get_env_info(device))

    scale = int(cfg.get("scale", 4))
    degradation_cfgs = cfg.get("degradation", [])
    pipeline = DegradationPipeline(degradation_cfgs, scale=scale)
    degradation_tag = pipeline.tag

    # Build test sets
    test_cfgs = cfg.get("data", {}).get("test", [])
    test_sets = []
    for t_cfg in test_cfgs:
        ds_name = t_cfg["name"]
        ds_root = t_cfg["root"]
        max_imgs = t_cfg.get("max_images", None)

        ds_obj = DATASETS.build(
            "image_folder",
            root=ds_root,
            scale=scale,
            degradation=degradation_cfgs,
            mode="eval",
            max_images=max_imgs,
            seed=seed,
        )
        test_sets.append({"name": ds_name, "dataset": ds_obj})

    eval_cfg = cfg.get("evaluation", {})
    metric_names_and_params = eval_cfg.get(
        "metrics",
        [{"name": "psnr", "y_channel": True}, {"name": "ssim", "y_channel": True}],
    )
    save_samples = int(eval_cfg.get("save_samples", 5))

    all_results_rows = []
    summary_models_info = []

    # Case A: Classical Baselines experiment (models list)
    if "models" in cfg and isinstance(cfg["models"], list):
        if eval_only and checkpoint:
            logger.warning("Checkpoint provided for classical baselines evaluation, ignoring.")

        for m_entry in cfg["models"]:
            m_name = m_entry["name"]
            m_params = m_entry.get("params", {})
            m_label = m_entry.get("label", m_name)

            model = MODELS.build(m_name, scale=scale, **m_params)
            param_cnt = count_parameters(model)

            logger.info(f"Evaluating model: {m_label}")
            rows = evaluate_model(
                model=model,
                model_label=m_label,
                test_sets=test_sets,
                metric_names_and_params=metric_names_and_params,
                scale=scale,
                device=device,
                out_dir=output_dir,
                save_samples=save_samples,
                degradation_tag=degradation_tag,
            )
            all_results_rows.extend(rows)

            summary_models_info.append({"label": m_label, "params": param_cnt})

    # Case B: Neural Network / Single Model experiment
    elif "model" in cfg and isinstance(cfg["model"], dict):
        m_cfg = cfg["model"]
        m_name = m_cfg["name"]
        m_params = m_cfg.get("params", {})
        m_label = m_cfg.get("label", m_name)

        model = MODELS.build(m_name, scale=scale, **m_params)
        param_cnt = count_parameters(model)

        if eval_only:
            if not checkpoint:
                raise ValueError(
                    "eval_only=True requires a path to a checkpoint file, but none was provided."
                )
            ckpt_path = Path(checkpoint)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
            logger.info(f"Loading checkpoint weights from: {ckpt_path}")
            ckpt_data = torch.load(ckpt_path, map_location=device)
            state_dict = ckpt_data.get("model_state", ckpt_data)
            model.load_state_dict(state_dict)

        else:
            # Training phase
            train_cfg = cfg.get("data", {}).get("train")
            val_cfg = cfg.get("data", {}).get("val")

            if not train_cfg or not val_cfg:
                raise ValueError(
                    "Training config requires data.train and data.val sections."
                )

            loader_cfg = cfg.get("data", {}).get("loader", {})
            batch_size = int(loader_cfg.get("batch_size", 32))
            num_workers = int(loader_cfg.get("num_workers", 2))

            train_ds = DATASETS.build(
                "image_folder",
                root=train_cfg["root"],
                scale=scale,
                degradation=degradation_cfgs,
                mode="train",
                patch_size=train_cfg.get("patch_size", 96),
                augment=train_cfg.get("augment", True),
                repeat=train_cfg.get("repeat", 10),
                max_images=train_cfg.get("max_images", None),
                seed=seed,
            )

            val_ds = DATASETS.build(
                "image_folder",
                root=val_cfg["root"],
                scale=scale,
                degradation=degradation_cfgs,
                mode="eval",
                max_images=val_cfg.get("max_images", None),
                seed=seed,
            )

            train_loader = DataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=(device.type == "cuda"),
            )

            val_loader = DataLoader(
                val_ds,
                batch_size=1,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=(device.type == "cuda"),
            )

            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                cfg=cfg,
                device=device,
                out_dir=output_dir,
                logger=logger,
            )
            trainer.fit()

            # Load best checkpoint for final evaluation
            best_ckpt = output_dir / "checkpoints" / "best.pth"
            if best_ckpt.exists():
                logger.info(f"Loading best checkpoint for evaluation: {best_ckpt}")
                ckpt_data = torch.load(best_ckpt, map_location=device)
                model.load_state_dict(ckpt_data["model_state"])

        logger.info(f"Evaluating model: {m_label}")
        rows = evaluate_model(
            model=model,
            model_label=m_label,
            test_sets=test_sets,
            metric_names_and_params=metric_names_and_params,
            scale=scale,
            device=device,
            out_dir=output_dir,
            save_samples=save_samples,
            degradation_tag=degradation_tag,
        )
        all_results_rows.extend(rows)
        summary_models_info.append({"label": m_label, "params": param_cnt})

    else:
        raise ValueError("Config must contain either a 'model' dict or 'models' list.")

    # Write metrics.csv
    df_metrics = pd.DataFrame(all_results_rows)
    df_metrics.to_csv(metrics_csv, index=False)
    logger.info(f"Metrics saved to: {metrics_csv}")

    # Build and write summary.json
    summary_data = {
        "experiment_name": exp_name,
        "scale": scale,
        "degradation": degradation_tag,
        "summary": [],
    }

    metric_keys = [m["name"] for m in metric_names_and_params]

    for model_info in summary_models_info:
        m_label = model_info["label"]
        p_cnt = model_info["params"]

        for t_info in test_sets:
            ds_name = t_info["name"]
            sub_df = df_metrics[
                (df_metrics["model"] == m_label) & (df_metrics["dataset"] == ds_name)
            ]

            if not sub_df.empty:
                metrics_summary = {}
                for mk in metric_keys:
                    if mk in sub_df.columns:
                        metrics_summary[mk] = float(sub_df[mk].mean())

                avg_inf_time = float(sub_df["inference_time_s"].mean())
                num_images = len(sub_df)

                summary_data["summary"].append(
                    {
                        "model": m_label,
                        "dataset": ds_name,
                        "n_images": num_images,
                        "param_count": p_cnt,
                        "mean_inference_time_s": avg_inf_time,
                        "metrics": metrics_summary,
                    }
                )

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    logger.info(f"Summary saved to: {summary_json}")
    logger.info(f"Experiment '{exp_name}' completed successfully.")

    return summary_data
