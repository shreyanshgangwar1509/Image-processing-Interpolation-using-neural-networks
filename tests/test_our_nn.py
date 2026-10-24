"""Unit tests for our_nn model architecture."""
from pathlib import Path
import sys
import pytest
import torch

# Insert repo root into sys.path
repo_root = Path(__file__).parent.parent.resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.evaluation.evaluator import count_parameters
import src.models.registry as model_reg
from src.models.our_nn import OurNN


def test_our_nn_shapes():
    """Test output shapes for scales 2, 3, 4 across default and ablation flags."""
    lr = torch.rand(1, 3, 16, 16)
    scales = [2, 3, 4]

    ablations = [
        # Full default
        {},
        # a0 plain
        {"bicubic_skip": False, "attention": False, "edge_branch": False},
        # a1 skip
        {"bicubic_skip": True, "attention": False, "edge_branch": False},
        # a2 attn
        {"bicubic_skip": True, "attention": True, "edge_branch": False},
    ]

    for scale in scales:
        for flags in ablations:
            model = OurNN(scale=scale, **flags)
            model.eval()
            with torch.no_grad():
                out = model(lr)
            expected_h, expected_w = 16 * scale, 16 * scale
            assert out.shape == (1, 3, expected_h, expected_w), (
                f"Scale {scale} with flags {flags} returned shape {out.shape}, expected (1, 3, {expected_h}, {expected_w})"
            )


def test_bicubic_skip_zero_init_equivalence():
    """Test that untrained model with bicubic_skip=True and zero_init_tail=True equals plain bicubic."""
    lr = torch.rand(2, 3, 16, 16)
    scale = 4

    our_model = OurNN(scale=scale, bicubic_skip=True, zero_init_tail=True)
    our_model.eval()

    bicubic_model_cls = model_reg.MODELS.get("bicubic")
    bicubic_model = bicubic_model_cls(scale=scale)
    bicubic_model.eval()

    with torch.no_grad():
        out_our = our_model(lr)
        out_bicubic = bicubic_model(lr)

    assert torch.allclose(out_our, out_bicubic, atol=1e-5), (
        "Untrained OurNN with zero_init_tail and bicubic_skip should match plain bicubic output."
    )


def test_edge_kernel_buffers():
    """Test that edge kernel tensors are registered as buffers, not in model.parameters()."""
    model = OurNN(scale=4, edge_branch=True)

    buffer_names = dict(model.named_buffers()).keys()
    assert "sobel_x" in buffer_names
    assert "sobel_y" in buffer_names
    assert "laplacian" in buffer_names

    param_tensors = list(model.parameters())
    for p in param_tensors:
        assert p.requires_grad is True

    # Assert sobel_x is not in parameters
    param_ids = [id(p) for p in model.parameters()]
    assert id(model.sobel_x) not in param_ids
    assert id(model.sobel_y) not in param_ids
    assert id(model.laplacian) not in param_ids


def test_parameter_count_below_400k():
    """Test that parameter count of default OurNN at scale 4 is below 400,000."""
    model = OurNN(scale=4, n_feats=32, n_blocks=6)
    num_params = count_parameters(model)
    print(f"\n[INFO] OurNN scale=4 parameter count: {num_params}")
    assert num_params < 400000, f"Parameter count {num_params} exceeds 400,000 threshold."


def test_unsupported_scale_raises():
    """Test that an unsupported scale raises ValueError."""
    with pytest.raises(ValueError) as exc_info:
        OurNN(scale=5)
    assert "supports scales 2, 3, 4" in str(exc_info.value)


def test_single_training_step():
    """Test single training step (forward, loss, backward, grad check)."""
    model = OurNN(scale=2, n_feats=16, n_blocks=2)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.L1Loss()

    lr = torch.rand(2, 3, 16, 16)
    hr = torch.rand(2, 3, 32, 32)

    optimizer.zero_grad()
    sr = model(lr)
    loss = criterion(sr, hr)
    loss.backward()

    assert torch.isfinite(loss), "Training loss must be finite."
    assert model.head.weight.grad is not None, "Head conv gradient should not be None."
    optimizer.step()
