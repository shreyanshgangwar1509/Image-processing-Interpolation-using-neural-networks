"""Services for model discovery, image degradation, inference execution, and zip export."""
import io
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import zipfile

import numpy as np
import pandas as pd
from PIL import Image, ImageChops, ImageOps
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from src.data.degradation import DegradationPipeline
from src.evaluation.evaluator import count_parameters
from src.metrics import METRICS
from src.models import MODELS
from src.utils.config import load_config
from src.utils.device import get_device

logger = logging.getLogger("sr_project.app")


def discover_eligible_models(repo_root: Path) -> List[Dict[str, Any]]:
    """Scans results/ for eligible neural network model checkpoints."""
    results_dir = repo_root / "results"
    if not results_dir.exists():
        return []

    eligible_models = []

    for exp_dir in sorted(results_dir.iterdir()):
        if not exp_dir.is_dir():
            continue

        exp_name = exp_dir.name
        # Skip smoke test runs
        if "smoke_test" in exp_name.lower():
            continue

        ckpt_path = exp_dir / "checkpoints" / "best.pth"
        cfg_path = exp_dir / "config_resolved.yaml"

        if not ckpt_path.exists() or not cfg_path.exists():
            continue

        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}

            # Classical baselines have 'models' list, single neural models have 'model' dict
            if "model" not in cfg or not isinstance(cfg["model"], dict):
                continue

            model_cfg = cfg["model"]
            model_name = model_cfg.get("name")
            if not model_name:
                continue

            scale = int(cfg.get("scale", 4))
            params = model_cfg.get("params", {})
            m_label = model_cfg.get("label", exp_name)

            # Try building model to verify validity
            try:
                model_inst = MODELS.build(model_name, scale=scale, **params)
                param_count = count_parameters(model_inst)
            except Exception as build_err:
                logger.warning(f"Skipping '{exp_name}': failed to build model '{model_name}': {build_err}")
                continue

            # Extract degradation tag
            degradation_cfgs = cfg.get("degradation", [])
            pipeline = DegradationPipeline(degradation_cfgs, scale=scale)
            degradation_tag = pipeline.tag

            eligible_models.append(
                {
                    "id": exp_name,
                    "label": m_label,
                    "experiment_name": exp_name,
                    "model_name": model_name,
                    "scale": scale,
                    "degradation": degradation_tag,
                    "param_count": param_count,
                    "checkpoint_path": str(ckpt_path.relative_to(repo_root)),
                    "config_path": str(cfg_path.relative_to(repo_root)),
                }
            )

        except Exception as e:
            logger.warning(f"Error parsing experiment folder '{exp_name}': {e}")
            continue

    return eligible_models


def load_model_from_checkpoint(
    exp_info: Dict[str, Any], device: torch.device, repo_root: Path
) -> nn.Module:
    """Loads a model instance with weights from its best.pth checkpoint."""
    cfg_path = repo_root / exp_info["config_path"]
    ckpt_path = repo_root / exp_info["checkpoint_path"]

    cfg = load_config(cfg_path)
    model_cfg = cfg["model"]
    model_name = model_cfg["name"]
    scale = int(cfg.get("scale", 4))
    params = model_cfg.get("params", {})

    model = MODELS.build(model_name, scale=scale, **params)

    ckpt_data = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt_data.get("model_state", ckpt_data)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model


def pil_to_tensor(pil_img: Image.Image) -> torch.Tensor:
    """Converts a PIL Image (RGB) to float32 Tensor (3, H, W) in range [0, 1]."""
    pil_rgb = pil_img.convert("RGB")
    arr = np.array(pil_rgb, dtype=np.float32) / 255.0  # (H, W, 3)
    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)
    return tensor


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Converts float32 Tensor (3, H, W) or (1, 3, H, W) in range [0, 1] to PIL Image."""
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
    arr = (tensor.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def apply_degradation_to_pil(
    hr_pil: Image.Image, scale: int, add_noise: bool = False, noise_sigma: float = 15.0
) -> Tuple[Image.Image, Image.Image]:
    """Crops HR so dimensions are divisible by scale, degrades to LR, returns (LR_pil, HR_cropped_pil)."""
    hr_tensor = pil_to_tensor(hr_pil)
    c, h, w = hr_tensor.shape

    crop_h = (h // scale) * scale
    crop_w = (w // scale) * scale

    if crop_h == 0 or crop_w == 0:
        raise ValueError(f"Image dimensions ({w}x{h}) are too small for scale x{scale}.")

    hr_cropped = hr_tensor[:, :crop_h, :crop_w]

    steps = [{"name": "bicubic_downscale", "scale": scale}]
    if add_noise:
        steps.append({"name": "gaussian_noise", "sigma": noise_sigma})

    pipeline = DegradationPipeline(steps, scale=scale)
    lr_tensor = pipeline(hr_cropped, generator=None)

    return tensor_to_pil(lr_tensor), tensor_to_pil(hr_cropped)


def generate_diff_map(sr_pil: Image.Image, hr_pil: Image.Image) -> Image.Image:
    """Generates a visual 3x normalized absolute difference map between SR and HR."""
    sr_arr = np.array(sr_pil.convert("RGB"), dtype=np.float32)
    hr_arr = np.array(hr_pil.convert("RGB"), dtype=np.float32)

    if sr_arr.shape != hr_arr.shape:
        # Resize SR to match HR if slightly off
        sr_pil_res = sr_pil.resize((hr_pil.width, hr_pil.height), Image.BICUBIC)
        sr_arr = np.array(sr_pil_res.convert("RGB"), dtype=np.float32)

    diff = np.abs(sr_arr - hr_arr) * 3.0  # Amplify difference for visual clarity
    diff = np.clip(diff, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(diff, mode="RGB")


def compute_metrics(
    sr_tensor: torch.Tensor,
    hr_tensor: torch.Tensor,
    scale: int,
    check_lpips: bool = True,
) -> Dict[str, Optional[float]]:
    """Calculates PSNR, SSIM, and optional LPIPS metrics using project conventions."""
    psnr_fn = METRICS.build("psnr", y_channel=True)
    ssim_fn = METRICS.build("ssim", y_channel=True)

    # Ensure batch dimension
    if sr_tensor.ndim == 3:
        sr_tensor = sr_tensor.unsqueeze(0)
    if hr_tensor.ndim == 3:
        hr_tensor = hr_tensor.unsqueeze(0)

    # Match dimensions if slightly mismatched
    if sr_tensor.shape != hr_tensor.shape:
        sr_tensor = F.interpolate(
            sr_tensor, size=(hr_tensor.shape[2], hr_tensor.shape[3]), mode="bicubic", align_corners=False
        )

    psnr_val = psnr_fn(sr_tensor, hr_tensor, scale=scale)
    ssim_val = ssim_fn(sr_tensor, hr_tensor, scale=scale)

    lpips_val = None
    if check_lpips:
        try:
            lpips_fn = METRICS.build("lpips")
            lpips_val = lpips_fn(sr_tensor, hr_tensor, scale=scale)
        except (ImportError, Exception) as e:
            logger.debug(f"LPIPS unavailable: {e}")

    return {
        "psnr": float(psnr_val),
        "ssim": float(ssim_val),
        "lpips": float(lpips_val) if lpips_val is not None else None,
    }


def create_export_zip(
    input_pil: Image.Image,
    results: List[Dict[str, Any]],
    summary_dict: Dict[str, Any],
) -> bytes:
    """Creates a ZIP archive bytes buffer containing output PNGs and summary CSV/JSON."""
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        # Save input image
        input_bytes = io.BytesIO()
        input_pil.save(input_bytes, format="PNG")
        zip_file.writestr("input_image.png", input_bytes.getvalue())

        # Save model outputs
        for res in results:
            m_label = res["model_label"]
            sr_bytes = io.BytesIO()
            res["sr_pil"].save(sr_bytes, format="PNG")
            zip_file.writestr(f"outputs/{m_label}_sr.png", sr_bytes.getvalue())

            if "diff_pil" in res and res["diff_pil"] is not None:
                diff_bytes = io.BytesIO()
                res["diff_pil"].save(diff_bytes, format="PNG")
                zip_file.writestr(f"outputs/{m_label}_diff_map.png", diff_bytes.getvalue())

        # Save summary JSON
        zip_file.writestr("summary.json", json.dumps(summary_dict, indent=2))

        # Save summary CSV
        rows = []
        for res in results:
            m_info = res["metrics"]
            rows.append(
                {
                    "model": res["model_label"],
                    "scale": res["scale"],
                    "inference_time_s": res["inference_time_s"],
                    "output_width": res["width"],
                    "output_height": res["height"],
                    "psnr": m_info.get("psnr"),
                    "ssim": m_info.get("ssim"),
                    "lpips": m_info.get("lpips"),
                }
            )
        df = pd.DataFrame(rows)
        zip_file.writestr("summary.csv", df.to_csv(index=False))

    return zip_buffer.getvalue()
