"""Unit tests for models, degradation, metrics, registries, and datasets."""
from pathlib import Path
import sys
import tempfile
import numpy as np
from PIL import Image
import pytest
import torch

# Insert repo root into sys.path
repo_root = Path(__file__).parent.parent.resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.data.datasets import ImageFolderDataset
from src.data.degradation import BicubicDownscale, DegradationPipeline, GaussianNoise
import src.data.registry as data_reg
import src.models.registry as model_reg
import src.metrics.registry as metric_reg
from src.utils.registry import Registry


def test_models_shapes():
    """Check that all registered models output shape == input shape * scale."""
    lr = torch.rand(1, 3, 16, 16)
    scales = [2, 4]

    for name in model_reg.MODELS.names():
        model_cls = model_reg.MODELS.get(name)
        for scale in scales:
            model = model_cls(scale=scale)
            model.eval()
            with torch.no_grad():
                out = model(lr)
            assert out.shape == (1, 3, 16 * scale, 16 * scale), (
                f"Model '{name}' at scale {scale} returned shape {out.shape}, expected (1, 3, {16*scale}, {16*scale})"
            )

    # Test edsr_lite specifically at scale 3
    edsr_cls = model_reg.MODELS.get("edsr_lite")
    edsr3 = edsr_cls(scale=3)
    edsr3.eval()
    with torch.no_grad():
        out3 = edsr3(lr)
    assert out3.shape == (1, 3, 48, 48)


def test_degradation():
    """Test degradation step shapes, deterministic noise seeding, and pipeline tag."""
    img = torch.rand(3, 32, 32)
    downscale_step = BicubicDownscale(scale=2)
    out_down = downscale_step(img)
    assert out_down.shape == (3, 16, 16)

    # Gaussian noise determinism
    noise_step = GaussianNoise(sigma=15)
    gen1 = torch.Generator().manual_seed(123)
    gen2 = torch.Generator().manual_seed(123)
    gen3 = torch.Generator().manual_seed(456)

    out_noise1 = noise_step(img, generator=gen1)
    out_noise2 = noise_step(img, generator=gen2)
    out_noise3 = noise_step(img, generator=gen3)

    assert torch.allclose(out_noise1, out_noise2), "Same seed should yield identical noise output."
    assert not torch.allclose(out_noise1, out_noise3), "Different seeds should yield different noise output."

    # Pipeline tag
    pipe = DegradationPipeline(
        [{"name": "gaussian_blur", "sigma": 1.0}, {"name": "bicubic_downscale"}, {"name": "gaussian_noise", "sigma": 15}],
        scale=4,
    )
    assert pipe.tag == "gaussian_blur_s1+bicubic_down_x4+gaussian_noise_s15"


def test_metrics():
    """Test PSNR and SSIM metric properties."""
    img = (torch.rand(1, 3, 32, 32) * 255.0).round() / 255.0
    psnr_fn = metric_reg.METRICS.build("psnr", y_channel=True)
    ssim_fn = metric_reg.METRICS.build("ssim", y_channel=True)

    # Identical images
    psnr_clean = psnr_fn(img, img, scale=2)
    ssim_clean = ssim_fn(img, img, scale=2)
    assert psnr_clean == 100.0, f"Expected PSNR 100.0 for identical images, got {psnr_clean}"
    assert pytest.approx(ssim_clean, abs=1e-4) == 1.0, f"Expected SSIM ~1.0 for identical images, got {ssim_clean}"

    # Noisy comparison: slightly noisy vs more noisy
    gen1 = torch.Generator().manual_seed(1)
    gen2 = torch.Generator().manual_seed(2)

    slightly_noisy = (img + torch.randn(img.shape, generator=gen1) * (5.0 / 255.0)).clamp(0, 1)
    more_noisy = (img + torch.randn(img.shape, generator=gen2) * (30.0 / 255.0)).clamp(0, 1)

    psnr_slight = psnr_fn(slightly_noisy, img, scale=2)
    psnr_more = psnr_fn(more_noisy, img, scale=2)
    assert psnr_slight > psnr_more, f"Slightly noisy ({psnr_slight}) should have higher PSNR than more noisy ({psnr_more})"


def test_registry_errors():
    """Test KeyError on unknown name with available list and duplicate registration error."""
    reg = Registry("TEST")

    @reg.register("item1")
    def dummy_item1():
        pass

    with pytest.raises(KeyError) as exc_info:
        reg.get("unknown_item")
    err_str = str(exc_info.value)
    assert "Unknown name 'unknown_item'" in err_str
    assert "item1" in err_str

    with pytest.raises(KeyError):

        @reg.register("item1")
        def duplicate_item():
            pass


def test_dataset_train_eval():
    """Test image_folder dataset in train and eval modes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        for i in range(3):
            arr = (np.random.rand(128, 128, 3) * 255).astype(np.uint8)
            img = Image.fromarray(arr)
            img.save(tmp_path / f"img_{i}.png")

        degrad_cfg = [{"name": "bicubic_downscale"}]

        # Train dataset
        ds_train = ImageFolderDataset(
            root=tmp_path,
            scale=2,
            degradation=degrad_cfg,
            mode="train",
            patch_size=48,
            augment=True,
            repeat=2,
        )
        assert len(ds_train) == 6
        item_train = ds_train[0]
        assert item_train["hr"].shape == (3, 48, 48)
        assert item_train["lr"].shape == (3, 24, 24)

        # Eval dataset
        ds_eval = ImageFolderDataset(
            root=tmp_path,
            scale=2,
            degradation=degrad_cfg,
            mode="eval",
            seed=42,
        )
        assert len(ds_eval) == 3
        item_eval = ds_eval[0]
        assert item_eval["hr"].shape == (3, 128, 128)
        assert item_eval["lr"].shape == (3, 64, 64)
