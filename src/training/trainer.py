"""Trainer implementation for super-resolution neural network models."""
import csv
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.losses import LOSSES
from src.metrics import METRICS


class Trainer:
    """Trainer class for managing model training and validation."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: Dict[str, Any],
        device: torch.device,
        out_dir: Path,
        logger: logging.Logger,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device
        self.out_dir = Path(out_dir)
        self.logger = logger

        self.ckpt_dir = self.out_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_csv_path = self.out_dir / "train_log.csv"

        # Loss function
        loss_cfg = cfg.get("loss", {"name": "l1"})
        loss_name = loss_cfg["name"]
        loss_params = loss_cfg.get("params", {})
        self.criterion = LOSSES.build(loss_name, **loss_params).to(device)

        # Optimizer
        opt_cfg = cfg.get("optimizer", {"name": "adam", "lr": 1e-4})
        opt_name = opt_cfg["name"].lower()
        lr = float(opt_cfg.get("lr", 1e-4))
        weight_decay = float(opt_cfg.get("weight_decay", 0.0))

        if opt_name == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif opt_name == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer '{opt_name}'. Expected 'adam' or 'adamw'.")

        # Scheduler
        sched_cfg = cfg.get("scheduler", {"name": "none"})
        sched_name = sched_cfg.get("name", "none").lower()
        if sched_name == "step":
            step_size = int(sched_cfg.get("step_size", 40))
            gamma = float(sched_cfg.get("gamma", 0.5))
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=step_size, gamma=gamma
            )
        else:
            self.scheduler = None

        # Metrics for validation
        self.val_metrics = {
            "psnr": METRICS.build("psnr", y_channel=True),
            "ssim": METRICS.build("ssim", y_channel=True),
        }

        # Setup train_log.csv if not exists
        if not self.log_csv_path.exists():
            with open(self.log_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["epoch", "train_loss", "val_psnr", "val_ssim", "lr", "epoch_time_s"]
                )

    def fit(self) -> Dict[str, Any]:
        """Runs the training loop and returns training stats."""
        training_cfg = self.cfg.get("training", {})
        epochs = int(training_cfg.get("epochs", 100))
        val_every = int(training_cfg.get("val_every", 5))
        use_amp = bool(training_cfg.get("amp", False)) and self.device.type == "cuda"
        grad_clip = training_cfg.get("grad_clip", None)

        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        best_val_psnr = -1.0
        is_tty = sys.stdout.isatty()

        self.logger.info(f"Starting training for {epochs} epochs on device: {self.device}")

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            self.model.train()
            total_loss = 0.0
            num_batches = 0

            pbar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch}/{epochs}",
                disable=not is_tty,
                leave=False,
            )

            for batch in pbar:
                lr_t = batch["lr"].to(self.device)
                hr_t = batch["hr"].to(self.device)

                self.optimizer.zero_grad()

                if use_amp:
                    with torch.cuda.amp.autocast():
                        sr_t = self.model(lr_t)
                        loss = self.criterion(sr_t, hr_t)
                    scaler.scale(loss).backward()
                    if grad_clip is not None:
                        scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), float(grad_clip)
                        )
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    sr_t = self.model(lr_t)
                    loss = self.criterion(sr_t, hr_t)
                    loss.backward()
                    if grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), float(grad_clip)
                        )
                    self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1
                if is_tty:
                    pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            mean_train_loss = total_loss / max(num_batches, 1)

            if self.scheduler is not None:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]

            # Validation step
            val_psnr, val_ssim = None, None
            if epoch % val_every == 0 or epoch == epochs:
                val_psnr, val_ssim = self.evaluate_val()
                self.logger.info(
                    f"Epoch {epoch:03d}/{epochs:03d} - Train Loss: {mean_train_loss:.6f} - "
                    f"Val PSNR: {val_psnr:.4f} dB - Val SSIM: {val_ssim:.4f} - LR: {current_lr:.6e}"
                )

                if val_psnr > best_val_psnr:
                    best_val_psnr = val_psnr
                    self.save_checkpoint("best.pth", epoch, val_psnr)
            else:
                self.logger.info(
                    f"Epoch {epoch:03d}/{epochs:03d} - Train Loss: {mean_train_loss:.6f} - LR: {current_lr:.6e}"
                )

            epoch_time = time.time() - t0

            # Save last checkpoint
            self.save_checkpoint("last.pth", epoch, val_psnr if val_psnr is not None else -1.0)

            # Append to log CSV incrementally
            with open(self.log_csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        epoch,
                        f"{mean_train_loss:.6f}",
                        f"{val_psnr:.4f}" if val_psnr is not None else "",
                        f"{val_ssim:.4f}" if val_ssim is not None else "",
                        f"{current_lr:.6e}",
                        f"{epoch_time:.2f}",
                    ]
                )

        return {"best_val_psnr": best_val_psnr}

    @torch.no_grad()
    def evaluate_val(self) -> Tuple[float, float]:
        """Runs validation evaluation over val_loader."""
        self.model.eval()
        scale = int(self.cfg.get("scale", 2))

        psnr_list = []
        ssim_list = []

        for batch in self.val_loader:
            lr_t = batch["lr"].to(self.device)
            hr_t = batch["hr"].to(self.device)

            sr_t = self.model(lr_t).clamp(0.0, 1.0)

            # batch size is usually 1 for val, but support B >= 1
            for i in range(sr_t.shape[0]):
                sr_single = sr_t[i : i + 1]
                hr_single = hr_t[i : i + 1]
                psnr_val = self.val_metrics["psnr"](sr_single, hr_single, scale=scale)
                ssim_val = self.val_metrics["ssim"](sr_single, hr_single, scale=scale)
                psnr_list.append(psnr_val)
                ssim_list.append(ssim_val)

        mean_psnr = sum(psnr_list) / max(len(psnr_list), 1)
        mean_ssim = sum(ssim_list) / max(len(ssim_list), 1)
        return mean_psnr, mean_ssim

    def save_checkpoint(self, filename: str, epoch: int, val_psnr: float) -> None:
        """Saves checkpoint dictionary to checkpoints/ directory."""
        model_cfg = self.cfg.get("model", {})
        ckpt = {
            "model_state": self.model.state_dict(),
            "epoch": epoch,
            "val_psnr": val_psnr,
            "model_name": model_cfg.get("name", "unknown"),
            "model_params": model_cfg.get("params", {}),
            "scale": self.cfg.get("scale", 2),
        }
        torch.save(ckpt, self.ckpt_dir / filename)
