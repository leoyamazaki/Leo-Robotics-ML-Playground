import numpy as np
import pytest
import torch

from simple_arm.contact_learning import make_model, training_arrays
from simple_arm.one_step_stationary import (Config, displacement_metrics, objective,
                                            recommendation, run, stationary_intervals, train)


def fixture_data():
    states = np.zeros((6, 201, 13))
    states[..., 8] = 1
    states[0, 5:, 4] = .0002  # retain small ground-truth displacement
    data = {"states": states, "actions": np.zeros((6, 200, 2)),
            "contacts": np.zeros((6, 200, 10), dtype=bool),
            "split": np.repeat([0, 1, 2], 2), "kinds": np.zeros(6, dtype=int),
            "metadata": {"seed": 9, "lengths": [.3, .25],
                         "kinds": ["overhead", "near_miss", "light_contact", "clear_push"]}}
    data["contacts"][0, :5] = True
    x, y, _ = training_arrays(data, 5, 0)
    torch.manual_seed(5)
    model = make_model(8)
    checkpoint = {"weights": model.state_dict(), "hidden_width": 8, "stride": 5,
                  "metadata": data["metadata"], "input_mean": x.mean(0), "target_mean": y.mean(0),
                  "input_scale": torch.ones(15), "target_scale": torch.ones(12)}
    return data, checkpoint


def test_exact_one_x_and_analytic_gradient_coefficients():
    torch.manual_seed(11)
    prediction = torch.randn(4, 12, dtype=torch.float64, requires_grad=True)
    target = torch.randn_like(prediction)
    contacts, still = torch.tensor([1., 5., 1., 5.]), torch.tensor([True, True, False, False])
    original = ((prediction - target).square().mean(-1) * contacts).mean()
    one = objective(prediction, target, contacts, still, 1)
    assert torch.equal(original, one)
    original_grad, = torch.autograd.grad(original, prediction)
    for weight in (1, 5, 20, 100):
        loss = objective(prediction, target, contacts, still, weight)
        gradient, = torch.autograd.grad(loss, prediction)
        coefficients = torch.ones_like(prediction)
        coefficients[:2, 4:6] = weight
        torch.testing.assert_close(gradient, original_grad * coefficients, rtol=1e-14, atol=1e-14)
        analytic = 2 * (prediction - target) * contacts[:, None] * coefficients / (4 * 12)
        torch.testing.assert_close(gradient, analytic)
    with pytest.raises(ValueError):
        objective(prediction, target, contacts, still, 0)


def test_stationarity_start_following_samples_local_anchor_and_boundaries():
    states = np.zeros((1, 201, 13))
    states[0, 0, 9] = .005  # inclusive velocity boundary at actual start
    states[0, 1, 4] = .0005  # inclusive excursion boundary
    assert stationary_intervals(states)[0, 0]
    states[0, 0, 9] = np.nextafter(.005, 1)
    assert not stationary_intervals(states)[0, 0]
    states[0, 0, 9] = 0
    states[0, 2, 10] = .006  # velocity transient despite no displacement
    assert not stationary_intervals(states)[0, 0]
    states[0, 2, 10] = 0
    states[0, 1, 4] = np.nextafter(.0005, 1)  # out and back excursion
    assert not stationary_intervals(states)[0, 0]
    states[:] = 0
    states[0, 6:, 4] = .03  # real push, then stationary at new actual location
    mask = stationary_intervals(states)
    assert mask[0, 0] and not mask[0, 1] and mask[0, 2:].all()
    states[0, 15, 9] = .02  # shared boundary belongs to both intervals
    mask = stationary_intervals(states)
    assert not mask[0, 2] and not mask[0, 3]
    assert stationary_intervals(states, .03, .04).all()


def test_matched_initialization_batches_updates_normalization_and_determinism(tmp_path, monkeypatch):
    data, reference = fixture_data()
    source = tmp_path / "source.pt"
    torch.save(reference, source)
    original_bytes = source.read_bytes()
    import mujoco
    monkeypatch.setattr(mujoco, "mj_step", lambda *a, **kw: pytest.fail("Training stepped physics"))
    # No recursive predictor may be called from the trainer or its diagnostics.
    monkeypatch.setattr("simple_arm.one_step_stationary.rollout", lambda *a, **kw: pytest.fail("Training used rollout"))
    monkeypatch.setattr("simple_arm.contact_learning.differentiable_next", lambda *a, **kw: pytest.fail("Recursive training"))
    reports, saved = [], []
    config = Config(epochs=2, batch_size=32)
    for name, weight in (("one", 1), ("five", 5), ("repeat", 5)):
        path = tmp_path / name
        reports.append(train(data, source, path, weight, config))
        final = torch.load(path / "final.pt", weights_only=True)
        saved.append(final)
        best = torch.load(path / "best.pt", weights_only=True)
        report = reports[-1]
        assert len(report["history"]) == 3
        assert report["history"][0]["updates"] == 0
        assert report["optimizer_updates"] == 6  # ceil(80/32)*2
        assert final["epoch"] == 2
        assert best["epoch"] == min(report["history"], key=lambda r: r["validation"]["unweighted"])["epoch"]
        for key in ("input_mean", "input_scale", "target_mean", "target_scale"):
            assert torch.equal(final[key], reference[key])
        with pytest.raises(FileExistsError):
            train(data, source, path, weight, config)
    assert len({r["batch_order_sha256"] for r in reports}) == 1
    assert len({r["source_sha256"] for r in reports}) == 1
    for r in reports[1:]:
        assert r["history"][0]["train"]["unweighted"] == reports[0]["history"][0]["train"]["unweighted"]
    for key in saved[1]["weights"]:
        assert torch.equal(saved[1]["weights"][key], saved[2]["weights"][key])
    assert source.read_bytes() == original_bytes
    assert (1736 * 40 + 255) // 256 * 12 == 3264


def test_false_motion_missed_motion_direction_and_zero_reference():
    data, _ = fixture_data()
    data["states"][0, 5:, 4] = .003
    actual = data["states"][:, ::5]
    teacher = actual[:, 1:].copy()
    teacher[1, 0, 4] += .0021
    teacher[0, 0, 4] = -.003  # equal magnitude, opposite direction
    report = displacement_metrics(data, teacher)
    assert report["rates"]["0.002"]["false_motion_count"] == 1
    assert report["rates"]["0.002"]["contact_motion_samples"] == 1
    assert report["rates"]["0.002"]["missed_motion_count"] == 0
    assert report["contact_motion"]["mean_angle_error_deg"] == pytest.approx(180)
    assert report["contact_motion"]["magnitude_rmse_m"] == pytest.approx(0)
    assert report["contact_motion"]["within_30deg_and_above_2mm_rate"] == 0
    assert report["cases"]["clear_push"] == {"samples": 0, "rmse_m": None}
    zero = displacement_metrics(data, actual[:, :-1].copy())
    assert zero["rates"]["0.002"]["false_motion_rate"] == 0
    assert zero["rates"]["0.002"]["missed_motion_rate"] == 1
    assert zero["contact_motion"]["direction_samples"] == 0
    assert zero["contact_motion"]["within_30deg_and_above_2mm_rate"] == 0


def test_epoch_zero_best_is_retained_and_final_is_primary(tmp_path, monkeypatch):
    data, reference = fixture_data()
    source = tmp_path / "source.pt"
    torch.save(reference, source)
    calls = []
    def diagnostic(model, arrays, stationary, multiplier):
        # Validation is always the second call in each epoch, and worsens.
        epoch = len(calls) // 2
        calls.append(epoch)
        return {"weighted_objective": 100 - epoch, "unweighted": 1. + epoch}
    monkeypatch.setattr("simple_arm.one_step_stationary.losses", diagnostic)
    report = train(data, source, tmp_path / "run", 5, Config(epochs=2))
    best = torch.load(tmp_path / "run/best.pt", weights_only=True)
    final = torch.load(tmp_path / "run/final.pt", weights_only=True)
    assert best["epoch"] == 0 and report["selected_best_epoch"] == 0
    assert final["epoch"] == 2 and report["primary_epoch"] == 2
    for key in reference["weights"]:
        assert torch.equal(best["weights"][key], reference["weights"][key])


def test_stationary_targets_retain_small_recorded_displacement():
    data, _ = fixture_data()
    _, y, _ = training_arrays(data, 5, 0)
    still = stationary_intervals(data["states"][data["split"] == 0]).reshape(-1)
    assert still[0] and y[0, 4] == pytest.approx(.0002)
    prediction = y.clone()
    assert objective(prediction, y, torch.ones(len(y)), torch.tensor(still), 100) == 0
    prediction[0, 4] = 0
    assert objective(prediction, y, torch.ones(len(y)), torch.tensor(still), 100) > 0


def metrics(stationary=1., moving=1., clear=1., missed=.2):
    one = {"stationary": {"rmse_m": stationary}, "moving_contact": {"rmse_m": moving},
           "cases": {"clear_push": {"rmse_m": clear}}, "rates": {"0.002": {"missed_motion_rate": missed}}}
    return {"one_step": one}


def model(weight, **kwargs):
    return {"multiplier": weight, "evaluation": {"boundary": metrics(**kwargs), "broad": metrics(**kwargs)}}


def test_recommendation_guards_ties_missing_cohorts_and_zero_failure():
    models = {"weight_1": model(1), "weight_5": model(5, stationary=.5),
              "weight_20": model(20, stationary=.5), "weight_100": model(100, stationary=.1, missed=1.)}
    result = recommendation(models)
    assert result["recommended"] == "weight_5"  # exact tie goes smaller
    assert not result["candidates"]["weight_100"]["eligible"]  # zero predictor tradeoff
    models["weight_5"]["evaluation"]["broad"] = metrics(stationary=1.1)
    models["weight_20"]["evaluation"]["boundary"] = metrics(stationary=.1, moving=1.10001)
    assert recommendation(models)["recommended"] == "weight_1"
    models["weight_5"] = model(5, stationary=.5, clear=None)
    assert recommendation(models)["recommended"] == "weight_1"
    models["weight_5"] = model(5, stationary=.5, moving=1.1, clear=1.1, missed=.25)
    assert recommendation(models)["recommended"] == "weight_5"  # inclusive limits
    assert recommendation(models)["deployment_endorsement"] is False


def test_overwrite_and_bad_configuration_refused(tmp_path):
    with pytest.raises(FileExistsError):
        run(tmp_path)
    for values in ({"epochs": 0}, {"speed_tolerance": -1}, {"excursion_tolerance": float("nan")}, {"learning_rate": 0}):
        with pytest.raises(ValueError):
            Config(**values)
    with pytest.raises(ValueError):
        run(tmp_path / "absent", multipliers=(5,))
