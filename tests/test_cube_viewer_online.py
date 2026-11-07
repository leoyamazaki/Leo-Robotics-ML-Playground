import numpy as np
import pytest
import torch

from simple_arm.contact_models import model_from_checkpoint
from simple_arm.cube_mpc import ModelAdapter
from simple_arm.cube_mpc_experiment import load_cases
from simple_arm.contact_learning import load_predictor
from test_residual_training import numerical_data, reference


def test_viewer_accepts_new_boundary_seed_without_changing_benchmark_guard(monkeypatch):
    states = np.zeros((160, 2, 13))
    states[..., 8] = 1
    states[:, -1, 4] = .05
    dataset = {"states": states, "kinds": np.tile(np.arange(4), 40),
               "metadata": {"seed": 592, "kinds": ["overhead", "near_miss", "light_contact", "clear_push"]}}
    monkeypatch.setattr("simple_arm.cube_mpc_experiment.load_boundary", lambda _: dataset)
    with pytest.raises(ValueError, match="292"):
        load_cases("unused")
    assert set(load_cases("unused", expected_seed=None)) == {23, 99, 139, 159}


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"))])
def test_device_adapter_agrees_with_cpu_predictor_and_preserves_caller_device(tmp_path, device):
    data = numerical_data()
    c = reference(data)
    c["weights"] = model_from_checkpoint(c).state_dict()
    path = tmp_path / "model.pt"
    torch.save(c, path)
    adapter = ModelAdapter(path, device=device)
    state, action = data["states"][2, :3], data["actions"][2, :3]
    actual = adapter(torch.tensor(state, dtype=torch.float32), torch.tensor(action, dtype=torch.float32))
    assert actual.device.type == "cpu"
    assert next(adapter.model.parameters()).device.type == device
    assert not any(p.requires_grad for p in adapter.model.parameters())
    np.testing.assert_allclose(actual.numpy(), load_predictor(path)(state, action), atol=1e-6)
