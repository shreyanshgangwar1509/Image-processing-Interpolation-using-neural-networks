"""Evaluation helper and model metrics calculation."""
from pathlib import Path
import time
from typing import Any, Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.metrics import METRICS
from src.utils.image_io import save_image


def count_parameters(model: nn.Module) -> int:
    """Counts the number of trainable parameters in a module."""
    if not getattr(model, "trainable", True):
        return 0
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_model(
    model: nn.Module,
    model_label: str,
    test_sets: List[Dict[str, Any]],
    metric_names_and_params: List[Dict[str, Any]],
    scale: int,
    device: torch.device,
    out_dir: Path,
    save_samples: int = 0,
    degradation_tag: str = "",
) -> List[Dict[str, Any]]:
    """Evaluates a model over a list of test sets and computes per-image metrics."""
    out_dir = Path(out_dir)
    samples_dir = out_dir / "samples"

    model = model.to(device)
    model.eval()

    # Build metric callables
    metric_objs = {}
    for m_cfg in metric_names_and_params:
        params = dict(m_cfg)
        name = params.pop("name")
        metric_objs[name] = METRICS.build(name, **params)

    results_rows = []

    for test_set_info in test_sets:
        dataset_name = test_set_info["name"]
        dataset = test_set_info["dataset"]

        set_samples_dir = samples_dir / dataset_name
        if save_samples > 0:
            set_samples_dir.mkdir(parents=True, exist_ok=True)

        for img_idx in range(len(dataset)):
            item = dataset[img_idx]
            lr_t = item["lr"].unsqueeze(0).to(device)  # (1, 3, H, W)
            hr_t = item["hr"].unsqueeze(0).to(device)  # (1, 3, H*scale, W*scale)
            img_name = item["name"]

            # Synchronize CUDA before timing if applicable
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()

            with torch.no_grad():
                sr_t = model(lr_t)

            if device.type == "cuda":
                torch.cuda.synchronize()
            inference_time = time.time() - t0

            sr_clamped = sr_t.clamp(0.0, 1.0)

            # Compute metrics
            row_metrics = {}
            for name, metric_fn in metric_objs.items():
                val = metric_fn(sr_clamped, hr_t, scale=scale)
                row_metrics[name] = val

            row = {
                "model": model_label,
                "dataset": dataset_name,
                "image": img_name,
                "scale": scale,
                "degradation": degradation_tag,
                **row_metrics,
                "inference_time_s": inference_time,
            }
            results_rows.append(row)

            # Save sample images if requested
            if img_idx < save_samples:
                # Save SR image
                sr_save_path = set_samples_dir / f"{img_name}_{model_label}_sr.png"
                save_image(sr_clamped.squeeze(0), sr_save_path)

                # Save HR and LR images once per dataset image
                hr_save_path = set_samples_dir / f"{img_name}_hr.png"
                lr_save_path = set_samples_dir / f"{img_name}_lr.png"

                if not hr_save_path.exists():
                    save_image(hr_t.squeeze(0), hr_save_path)

                if not lr_save_path.exists():
                    # Nearest neighbor upscaling of LR so it is viewable at HR size
                    lr_upscaled = F.interpolate(lr_t, scale_factor=scale, mode="nearest")
                    save_image(lr_upscaled.squeeze(0), lr_save_path)

    return results_rows
