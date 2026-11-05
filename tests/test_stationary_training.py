import numpy as np
import pytest
import torch

from simple_arm.contact_learning import load_predictor, make_model
from simple_arm.contact_simulation import model_hash
from simple_arm.cube_mpc_experiment import file_hash
from simple_arm.stationary_experiment import prediction_metrics, run
from simple_arm.stationary_training import (TrainingConfig, recursive_loss, stationary_intervals,
                                            stationary_penalty, train)


def fixture_data():
    states = np.zeros((6, 201, 13))
    states[..., 8] = 1
    states[..., 4] = .4
    data = {"states": states, "actions": np.zeros((6, 200, 2)), "contacts": np.zeros((6, 200, 10), dtype=bool),
            "starting_contact": np.zeros((6, 200), dtype=bool), "split": np.repeat([0, 1, 2], 2),
            "kinds": np.zeros(6, dtype=int), "metadata": {"seed": 9, "model_hash": model_hash(), "lengths": [.3, .25],
                                                         "kinds": ["overhead", "near_miss", "light_contact", "clear_push"]}}
    model = make_model(8)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model[-1].bias[4] = .001
    checkpoint = {"weights": model.state_dict(), "hidden_width": 8, "stride": 5, "metadata": data["metadata"],
                  "input_mean": torch.zeros(15), "input_scale": torch.ones(15),
                  "target_mean": torch.zeros(12), "target_scale": torch.ones(12)}
    return data, checkpoint


def test_stationary_masks_follow_each_local_interval_and_detect_out_and_back():
    data, _ = fixture_data()
    states = data["states"][:1]
    states[:, 6:, 4] += .03
    mask = stationary_intervals(states, .002)
    assert mask[0, 0] and not mask[0, 1] and mask[0, 2:].all()
    # A 20 ms transient must not be called stationary just because it returns.
    states[:, 21, 4] += .01
    assert not stationary_intervals(states, .002)[0, 4]
    assert stationary_intervals(states, .02)[0, 4]


def test_stationary_loss_uses_truth_anchor_masks_and_backpropagates():
    prediction = torch.zeros(2, 13, requires_grad=True)
    actual_start = torch.zeros(2, 13)
    actual_start[0, 4] = .1  # position after a real previous push, not episode origin
    value = stationary_penalty(prediction, actual_start, torch.tensor([1., 0.]))
    assert value.item() == pytest.approx(.25)
    value.backward()
    assert prediction.grad[0, 4] < 0
    assert prediction.grad[1].abs().sum() == 0


def test_recursive_feedback_and_per_step_actual_anchors(monkeypatch):
    calls = []
    def predict(model, current, action, checkpoint):
        calls.append(current.clone())
        result = current.clone()
        result[:, 4] += .01
        return result
    monkeypatch.setattr("simple_arm.stationary_training.differentiable_next", predict)
    actual = torch.zeros(1, 4, 13)
    actual[:, 2:, 4] = .02
    loss, base, extra = recursive_loss(None, actual, torch.zeros(1, 3, 2), torch.zeros(1, 3),
                                      torch.tensor([[1., 0., 1.]]), {}, 1.)
    assert [c[0, 4].item() for c in calls] == pytest.approx([0, .01, .02])
    # Both stationary intervals are .01 from their own actual starting position.
    assert extra.item() == pytest.approx((.005 + 0 + .005) / 3)
    assert loss.item() == pytest.approx(base.item() + extra.item())


def test_equal_window_update_budgets_determinism_normalization_and_sources(tmp_path, monkeypatch):
    data, checkpoint = fixture_data()
    source = tmp_path / "source.pt"
    torch.save(checkpoint, source)
    digest = file_hash(source)
    config = TrainingConfig(epochs=1, batch_size=64)
    # Neither training nor validation is allowed to step a simulator.
    import mujoco
    monkeypatch.setattr(mujoco, "mj_step", lambda *a, **kw: pytest.fail("Physics used"))
    stats = []
    for name, horizon, weight in (("control", 10, 0.), ("stationary", 10, 1.), ("long", 20, 1.), ("repeat", 10, 1.)):
        output = tmp_path / name
        report = train(data, source, output, horizon, weight, config)
        stats.append(report)
        saved = torch.load(output / "model.pt", weights_only=True)
        for key in ("input_mean", "input_scale", "target_mean", "target_scale"):
            assert torch.equal(saved[key], checkpoint[key])
        assert saved["hidden_width"] == 8 and saved["metadata"] == data["metadata"]
        assert np.isfinite(load_predictor(output / "model.pt", 5)(data["states"][0, 0], [0, 0])).all()
        with pytest.raises(FileExistsError):
            train(data, source, output, horizon, weight, config)
    assert {r["optimizer_updates"] for r in stats} == {1}
    assert {r["windows_per_epoch"] for r in stats} == {42}
    assert stats[2]["supervised_window_steps"] == 2 * stats[1]["supervised_window_steps"]
    for a, b in zip(stats[1]["history"], stats[3]["history"]):
        assert a["validation_40_step"] == b["validation_40_step"]
    a = torch.load(tmp_path / "stationary/last.pt", weights_only=True)
    b = torch.load(tmp_path / "repeat/last.pt", weights_only=True)
    assert all(torch.equal(a["weights"][k], b["weights"][k]) for k in a["weights"])
    assert file_hash(source) == digest


def test_teacher_forced_ar_metrics_and_stationary_subsets():
    data, _ = fixture_data()
    actual = data["states"][:, ::5]
    teacher = actual[:, 1:].copy()
    teacher[..., 4] += .001
    ar = actual.copy()
    ar[..., 4] += np.arange(41) * .001
    accumulated = ar.copy()
    report = prediction_metrics(data, teacher, ar, accumulated)
    assert report["one_step_cases"]["no_contact"]["cube_displacement_rmse_m"] == pytest.approx(.001)
    assert report["one_step_cases"]["clear_push"]["cube_displacement_rmse_m"] is None
    drift = report["stationary_drift"]["overhead"]
    assert drift["autoregressive"]["mean_max_drift_m"] == pytest.approx(.04)
    assert drift["autoregressive"]["fraction_over_2mm"] == 1
    assert drift["teacher_forced_next"]["fraction_over_2mm"] == 0
    assert report["teacher_forced_vs_autoregressive"]["all"]["autoregressive_increment_rmse_m"] == pytest.approx(.001)
    assert report["push_4s"]["episodes"] == 0


def test_validation_and_overwrite_guards(tmp_path):
    with pytest.raises(FileExistsError):
        run(tmp_path)
    for config in ({"stationary_threshold": -1}, {"stationary_weight": float("nan")}, {"epochs": 0}):
        with pytest.raises(ValueError):
            TrainingConfig(**config)
    with pytest.raises(ValueError):
        stationary_intervals(np.zeros((1, 41, 13)), .002)
