"""Numerical fixtures only: these tests never collect or stand in for experiment data."""

import json

import numpy as np
import pytest
import torch

from simple_arm.contact_learning import (apply_delta, differentiable_next, features,
                                         load_predictor, make_model, training_arrays)
from simple_arm.contact_models import (INPUT_FLOOR, RESIDUAL_ARCHITECTURE, ResidualBlock,
                                      checkpoint_features, model_from_checkpoint)
from simple_arm.contact_simulation import link_lengths
from simple_arm.cube_mpc import ACTION_CONVENTION, STATE_CONVENTION, ModelAdapter
from simple_arm.residual_training import (Config, episode_arrays, fit_normalization, load_data,
                                         recursive_batch, recursive_loss, recursive_windows,
                                         run, validation_score)
from test_capacity_screen import fixture_arrays


def numerical_data():
    data, _ = fixture_arrays()
    data["metadata"].update(base_dt=.02, physics_dt=.002, state=STATE_CONVENTION,
                            action=ACTION_CONVENTION, kinds=["numerical_fixture"], lengths=link_lengths())
    data["starting_contact"] = np.zeros((3, 200), dtype=bool)
    data["kinds"] = np.zeros(3, dtype=int)
    data["group_ids"] = np.arange(3)
    data["contacts"][0, :5] = True
    return data


def save_fixture(path, data):
    np.savez_compressed(path, **{key: value for key, value in data.items() if key != "metadata"},
                        metadata=json.dumps(data["metadata"]))


def reference(data, residual=True):
    return dict(fit_normalization(data), stride=5, metadata=data["metadata"],
                architecture=RESIDUAL_ARCHITECTURE if residual else {"type": "plain_mlp", "width": 16, "layers": 3},
                input_coordinate_frame="base_relative")


def test_exact_architecture_shapes_residual_behavior_and_gradients():
    torch.set_num_threads(2)
    model = model_from_checkpoint({"architecture": RESIDUAL_ARCHITECTURE})
    assert sum(p.numel() for p in model.parameters()) == 4746252
    assert sum(isinstance(m, torch.nn.Linear) for m in model.modules()) == 10
    assert all(m.bias is not None for m in model.modules() if isinstance(m, torch.nn.Linear))
    for shape in ((2, 15), (2, 3, 15)):
        x = torch.randn(shape, requires_grad=True)
        result = model(x)
        assert result.shape == (*shape[:-1], 12)
        result.square().mean().backward()
        assert torch.isfinite(x.grad).all()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    block = ResidualBlock(8)
    x = torch.randn(3, 8)
    torch.testing.assert_close(block(x), torch.nn.functional.silu(x + block.linear2(torch.nn.functional.silu(block.linear1(x)))))
    with torch.no_grad():
        block.linear2.weight.zero_()
        block.linear2.bias.zero_()
    torch.testing.assert_close(block(x), torch.nn.functional.silu(x))
    assert sum(p.numel() for p in make_model(1536).parameters()) == 4764684


def test_coordinate_contract_numpy_tensor_and_mpc(tmp_path):
    data = numerical_data()
    c = reference(data)
    model = model_from_checkpoint(c).eval()
    c["weights"] = model.state_dict()
    path = tmp_path / "residual.pt"
    torch.save(c, path)
    state, action = data["states"][2, :3], data["actions"][2, :3]
    actual_features = checkpoint_features(state, action, c)
    np.testing.assert_array_equal(actual_features, np.concatenate([state, action], -1))
    assert not np.allclose(features(state, action, link_lengths())[..., 4:6], actual_features[..., 4:6])
    tensor = differentiable_next(model, torch.tensor(state, dtype=torch.float32), torch.tensor(action, dtype=torch.float32), c)
    np.testing.assert_allclose(load_predictor(path, 5)(state, action), tensor.detach().numpy(), atol=1e-6)
    torch.testing.assert_close(ModelAdapter(path)(torch.tensor(state, dtype=torch.float32), torch.tensor(action, dtype=torch.float32)), tensor)


def test_legacy_predictions_unchanged_and_unknown_contracts_rejected(tmp_path):
    data = numerical_data()
    c = dict(reference(data, False), weights=make_model(16).state_dict(), hidden_width=16)
    del c["architecture"], c["input_coordinate_frame"]
    path = tmp_path / "legacy.pt"
    torch.save(c, path)
    model = make_model(16).eval()
    model.load_state_dict(c["weights"])
    state, action = data["states"][2, :3], data["actions"][2, :3]
    old = state.copy()
    angle, total = state[..., 0], state[..., 0] + state[..., 1]
    old[..., 4] -= link_lengths()[0] * np.cos(angle) + link_lengths()[1] * np.cos(total)
    old[..., 5] -= link_lengths()[0] * np.sin(angle) + link_lengths()[1] * np.sin(total)
    x = torch.tensor(np.concatenate([old, action], -1), dtype=torch.float32)
    with torch.no_grad():
        delta = model((x - c["input_mean"]) / c["input_scale"]) * c["target_scale"] + c["target_mean"]
    np.testing.assert_array_equal(load_predictor(path)(state, action), apply_delta(state, delta.numpy()))
    for bad in ({"input_coordinate_frame": "world_magic"}, {"architecture": {"type": "other"}},
                {"architecture": dict(RESIDUAL_ARCHITECTURE, blocks=3)}):
        with pytest.raises(ValueError, match="Unknown"):
            model_from_checkpoint(dict(c, **bad))


def test_training_only_normalization_windows_contacts_and_split_guards(tmp_path):
    data = numerical_data()
    norm = fit_normalization(data)
    changed = dict(data, states=data["states"].copy())
    changed["states"][1:] += 100
    for key, value in fit_normalization(changed).items():
        torch.testing.assert_close(value, norm[key], rtol=0, atol=0)
    x, _, weights = training_arrays(data, 5, 0, "base_relative")
    torch.testing.assert_close(x.mean(0), norm["input_mean"])
    assert (norm["input_scale"] >= torch.tensor(INPUT_FLOOR)).all()
    assert weights[0] == 5 and (weights[1:] == 1).all()
    windows = recursive_windows(3)
    assert windows.shape == (93, 2)
    assert windows[:, 1].min() == 0 and windows[:, 1].max() == 30
    assert (windows[:, 1] + 10 <= 40).all()
    arrays = episode_arrays(data, 0)
    batch = recursive_batch(arrays, recursive_windows(1)[[0, -1]], "cpu")
    torch.testing.assert_close(batch[0][0], arrays[0][0, :11])
    torch.testing.assert_close(batch[0][1], arrays[0][0, 30:41])
    assert batch[2][0, 0] and not batch[2][1].any()
    data["group_ids"][:] = 0
    path = tmp_path / "cross_split.npz"
    save_fixture(path, data)
    with pytest.raises(ValueError, match="group crosses"):
        load_data(path)


def test_feedback_gradient_flow_and_recursive_contact_weight(monkeypatch):
    import simple_arm.residual_training as module
    parameter = torch.nn.Parameter(torch.tensor(.2))
    seen, predicted = [], []
    def next_state(model, current, action, checkpoint):
        seen.append(current)
        result = current + parameter
        result.retain_grad()
        predicted.append(result)
        return result
    monkeypatch.setattr(module, "differentiable_next", next_state)
    states = torch.zeros(2, 4, 13)
    actions = torch.zeros(2, 3, 2)
    no_contact = torch.zeros(2, 3, dtype=torch.bool)
    loss = recursive_loss(None, states, actions, no_contact, {})
    contact_loss = recursive_loss(None, states, actions, ~no_contact, {})
    torch.testing.assert_close(contact_loss, loss * 3)
    assert seen[1] is predicted[0] and seen[2] is predicted[1]
    # Loss only at the last prediction still differentiates through earlier feedback.
    predicted[2].square().mean().backward()
    assert predicted[0].grad is not None and predicted[0].grad.abs().sum() > 0
    assert parameter.grad is not None and parameter.grad > 0


def assert_same_weights(a, b):
    for key in a["weights"]:
        torch.testing.assert_close(a["weights"][key], b["weights"][key], rtol=0, atol=0)


def test_cpu_two_stages_resume_equivalence_and_best_preservation(tmp_path, monkeypatch, capsys):
    data = numerical_data()
    path = tmp_path / "fixture.npz"
    save_fixture(path, data)
    whole = tmp_path / "whole"
    run(path, whole, Config(2, 2), "cpu", evaluate=False)
    first = tmp_path / "first"
    run(path, first, Config(1, 0), "cpu", evaluate=False)
    second = tmp_path / "second"
    run(path, second, Config(2, 1), "cpu", first / "one_step/last.pt", evaluate=False)
    resumed = tmp_path / "resumed"
    run(path, resumed, Config(2, 2), "cpu", second, evaluate=False)
    load = lambda root, stage, name: torch.load(root / stage / name, weights_only=True, map_location="cpu")
    for stage in ("one_step", "recursive"):
        a, b = load(whole, stage, "last.pt"), load(resumed, stage, "last.pt")
        assert_same_weights(a, b)
        assert_same_weights(a["best_checkpoint"], b["best_checkpoint"])
        for key in ("input_mean", "input_scale", "target_mean", "target_scale"):
            torch.testing.assert_close(load(whole, "one_step", "model.pt")[key], b[key], rtol=0, atol=0)
        assert (resumed / stage / "loss.png").exists()
        for state in b["optimizer"]["state"].values():
            assert all(not isinstance(value, torch.Tensor) or value.device.type == "cpu" for value in state.values())
    predict = load_predictor(resumed / "recursive/model.pt", 5)
    np.testing.assert_array_equal(predict(data["states"][2, 0], data["actions"][2, 0]),
                                  load_predictor(whole / "recursive/model.pt", 5)(data["states"][2, 0], data["actions"][2, 0]))
    import simple_arm.residual_training as module
    monkeypatch.setattr(module, "validation_score", lambda *args: 1e30)
    worse = tmp_path / "worse"
    run(path, worse, Config(2, 3), "cpu", resumed, evaluate=False)
    assert_same_weights(load(resumed, "recursive", "model.pt"), load(worse, "recursive", "model.pt"))
    with pytest.raises(ValueError, match="batch_size"):
        run(path, tmp_path / "bad", Config(2, 3, 128), "cpu", resumed, evaluate=False)
    output = capsys.readouterr()
    assert "epoch=" not in output.out and "complete:" in output.out
    assert "\r" not in output.err
    from simple_arm.residual_evaluation import evaluate_run
    summary = evaluate_run(path, resumed, tmp_path / "evaluation")
    assert summary["models"]["one_step"]["evaluation"]["mixed_test"]["episodes"] == 1
    assert set(summary["models"]["recursive"]["latency"]["batch_sizes"]) == {"1", "256"}
    assert (tmp_path / "evaluation/REPORT.md").exists()
    assert (tmp_path / "evaluation/mixed_test/recursive/test_rollouts.npz").exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware required")
def test_cuda_training_and_cpu_continuation(tmp_path):
    path = tmp_path / "fixture.npz"
    save_fixture(path, numerical_data())
    gpu, cpu = tmp_path / "gpu", tmp_path / "cpu"
    run(path, gpu, Config(1, 1), "cuda", evaluate=False)
    run(path, cpu, Config(1, 2), "cpu", gpu, evaluate=False)
    saved = torch.load(cpu / "recursive/last.pt", weights_only=True)
    assert saved["epoch"] == 2 and all(w.device.type == "cpu" for w in saved["weights"].values())


def test_missing_artifacts_and_nonfinite_diagnostics(tmp_path):
    with pytest.raises(FileNotFoundError, match="original data/"):
        run(tmp_path / "missing.npz", tmp_path / "output", evaluate=False)
    model = make_model(16)
    c = reference(numerical_data(), False)
    with pytest.raises(RuntimeError, match="Nonfinite one_step validation"):
        validation_score(model, (torch.full((1, 15), float("nan")), torch.zeros(1, 12)), c, "one_step", "cpu", 256)
