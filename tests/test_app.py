"""Tests for the super-resolution web application."""
import asyncio
import io
from pathlib import Path
import sys

from PIL import Image
import pytest
from fastapi import HTTPException, UploadFile

repo_root = Path(__file__).parent.parent.resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from app.main import discover_eligible_models, run_inference


def test_interpolation_methods_are_available_without_checkpoints(tmp_path):
    models = discover_eligible_models(tmp_path)

    assert len(models) == 8
    assert {model["model_name"] for model in models} == {
        "nearest",
        "bilinear",
        "bicubic",
        "lanczos",
    }
    assert {model["scale"] for model in models} == {2, 4}
    assert all(model["type"] == "interpolation" for model in models)


def test_lr_inference_runs_interpolation_methods_at_their_scales(tmp_path, monkeypatch):
    models = discover_eligible_models(tmp_path)
    monkeypatch.setattr("app.main.REPO_ROOT", tmp_path)
    monkeypatch.setattr("app.main.discover_eligible_models", lambda _: models)

    image_buffer = io.BytesIO()
    Image.new("RGB", (6, 5), color="red").save(image_buffer, format="PNG")
    uploaded_file = UploadFile(
        filename="input.png", file=io.BytesIO(image_buffer.getvalue())
    )

    call_res = run_inference(
        file=uploaded_file,
        mode="lr_input",
        selected_models="interpolation_nearest_x2,interpolation_bilinear_x4",
        device_request="cpu",
        degradation_type="bicubic",
        noise_sigma=15.0,
        ref_file=None,
    )
    result = asyncio.run(call_res) if asyncio.iscoroutine(call_res) else call_res

    assert [(item["width"], item["height"]) for item in result["results"]] == [
        (12, 10),
        (24, 20),
    ]
    assert [item["model_name"] for item in result["results"]] == [
        "nearest",
        "bilinear",
    ]


def test_interpolation_methods_are_rejected_for_hr_reference_mode(tmp_path, monkeypatch):
    models = discover_eligible_models(tmp_path)
    monkeypatch.setattr("app.main.REPO_ROOT", tmp_path)
    monkeypatch.setattr("app.main.discover_eligible_models", lambda _: models)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            run_inference(
                file=UploadFile(filename="input.png", file=io.BytesIO()),
                mode="hr_reference",
                selected_models="interpolation_nearest_x2",
                device_request="cpu",
                degradation_type="bicubic",
                noise_sigma=15.0,
                ref_file=None,
            )
        )
    assert exc_info.value.detail == (
        "Standard interpolation methods are available only in LR input mode."
    )


def test_lr_inference_with_separate_reference_computes_metrics_across_different_scales(tmp_path, monkeypatch):
    models = discover_eligible_models(tmp_path)
    monkeypatch.setattr("app.main.REPO_ROOT", tmp_path)
    monkeypatch.setattr("app.main.discover_eligible_models", lambda _: models)

    # 10x10 LR input
    lr_buf = io.BytesIO()
    Image.new("RGB", (10, 10), color="blue").save(lr_buf, format="PNG")
    lr_file = UploadFile(filename="lr.png", file=io.BytesIO(lr_buf.getvalue()))

    # 40x40 HR reference (4x)
    hr_buf = io.BytesIO()
    Image.new("RGB", (40, 40), color="blue").save(hr_buf, format="PNG")
    ref_file = UploadFile(filename="hr_ref.png", file=io.BytesIO(hr_buf.getvalue()))

    call_res = run_inference(
        file=lr_file,
        mode="lr_input",
        selected_models="interpolation_bicubic_x2,interpolation_bicubic_x4",
        device_request="cpu",
        degradation_type="bicubic",
        noise_sigma=15.0,
        ref_file=ref_file,
    )
    result = asyncio.run(call_res) if asyncio.iscoroutine(call_res) else call_res

    assert result["has_reference"] is True
    assert len(result["results"]) == 2

    for r in result["results"]:
        assert r["model_type"] == "interpolation"
        assert "psnr" in r["metrics"]
        assert "ssim" in r["metrics"]
        assert r["metrics"]["psnr"] is not None
        assert r["metrics"]["ssim"] is not None
        assert r["diff_b64"] is not None


def test_lr_inference_without_reference_marks_has_reference_false(tmp_path, monkeypatch):
    models = discover_eligible_models(tmp_path)
    monkeypatch.setattr("app.main.REPO_ROOT", tmp_path)
    monkeypatch.setattr("app.main.discover_eligible_models", lambda _: models)

    lr_buf = io.BytesIO()
    Image.new("RGB", (10, 10), color="green").save(lr_buf, format="PNG")
    lr_file = UploadFile(filename="lr.png", file=io.BytesIO(lr_buf.getvalue()))

    call_res = run_inference(
        file=lr_file,
        mode="lr_input",
        selected_models="interpolation_bicubic_x2",
        device_request="cpu",
        degradation_type="bicubic",
        noise_sigma=15.0,
        ref_file=None,
    )
    result = asyncio.run(call_res) if asyncio.iscoroutine(call_res) else call_res

    assert result["has_reference"] is False
    res_0 = result["results"][0]
    assert res_0["model_type"] == "interpolation"
    assert res_0["metrics"] == {}
    assert res_0["diff_b64"] is None
