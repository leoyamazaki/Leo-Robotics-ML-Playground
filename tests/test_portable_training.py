import numpy as np
import pytest
import torch

from simple_arm.capacity_screen import prepare
from simple_arm.contact_learning import load_predictor
from simple_arm.portable_training import fit, select_device
from test_capacity_screen import fixture_arrays


def test_cpu_resume_matches_uninterrupted_adam(tmp_path):
    torch.set_num_threads(2)
    data, reference = fixture_arrays()
    train, val = prepare(data, reference)
    options = dict(width=16, batch_size=13, device="cpu", dataset_sha256="fixture")
    whole = fit(reference, train, val, tmp_path / "whole", epochs=2, **options)
    first = fit(reference, train, val, tmp_path / "first", epochs=1, **options)
    resumed = fit(reference, train, val, tmp_path / "resumed", epochs=1, resume=first, **options)
    a, b = (torch.load(path, weights_only=True) for path in (whole, resumed))
    for key in a["weights"]:
        torch.testing.assert_close(a["weights"][key], b["weights"][key], rtol=0, atol=0)
    assert b["epoch"] == 2
    assert (tmp_path / "resumed/model.pt").exists()
    assert np.isfinite(load_predictor(resumed, 5)(data["states"][2, 0], data["actions"][2, 0])).all()
    with pytest.raises(ValueError, match="dataset differs"):
        fit(reference, train, val, tmp_path / "bad", resume=first, dataset_sha256="changed")


def test_device_selection(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert select_device("auto").type == "cpu"
    with pytest.raises(RuntimeError, match="CUDA requested"):
        select_device("cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware required")
@pytest.mark.parametrize("first_device,second_device", [("cuda", "cpu"), ("cpu", "cuda")])
def test_checkpoint_can_continue_across_devices(tmp_path, first_device, second_device):
    data, reference = fixture_arrays()
    train, val = prepare(data, reference)
    gpu = fit(reference, train, val, tmp_path / "first", width=512, epochs=1, device=first_device)
    saved = torch.load(gpu, weights_only=True, map_location="cpu")
    assert all(value.device.type == "cpu" for value in saved["weights"].values())
    predictor = load_predictor(gpu, 5)
    assert np.isfinite(predictor(data["states"][2, 0], data["actions"][2, 0])).all()
    cpu = fit(reference, train, val, tmp_path / "second", epochs=1, device=second_device, resume=gpu)
    assert torch.load(cpu, weights_only=True)["epoch"] == 2
