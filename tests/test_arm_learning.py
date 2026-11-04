import numpy as np
import pytest
import torch

from simple_arm.arm_learning import (
    DT, baseline, collect, errors, evaluate, load_predictor, make_model,
    normalization, train, transitions,
)
from simple_arm import ArmSimulation


def test_broad_collection_covers_missing_states(tmp_path):
    dataset = collect(tmp_path / "broad.npz", episodes=40, steps=20, seed=73, profile="broad")
    repeated = collect(tmp_path / "repeated.npz", episodes=40, steps=20, seed=73, profile="broad")
    np.testing.assert_array_equal(dataset["states"], repeated["states"])
    np.testing.assert_array_equal(dataset["actions"], repeated["actions"])
    assert dataset["metadata"]["collection_profile"] == "broad"
    assert len(dataset["metadata"]["scenarios"]) == 8
    for split in range(3):
        assert set(dataset["scenarios"][dataset["split"] == split]) == set(range(8))
    assert np.any(np.abs(dataset["states"][:, 0, 2:]) > 1)
    limits = ArmSimulation().action_limits
    actions = dataset["actions"]
    assert np.all(actions >= limits[:, 0]) and np.all(actions <= limits[:, 1])
    assert np.any(np.isclose(np.abs(actions), limits[:, 1]))
    assert np.any(np.abs(dataset["states"][:, 0, :2]) > [1.2, 1.6])
    irregular = actions[dataset["scenarios"] >= 4]
    changes = np.any(np.diff(irregular, axis=1) != 0, axis=-1)
    assert np.any(changes[:, np.arange(1, 20) % 5 != 0])


def test_collection_timing_splits_and_repeatability(tmp_path):
    first = collect(tmp_path / "first.npz", episodes=20, steps=10)
    second = collect(tmp_path / "second.npz", episodes=20, steps=10)
    np.testing.assert_array_equal(first["states"], second["states"])
    np.testing.assert_array_equal(first["actions"], second["actions"])
    assert first["states"].shape == (20, 11, 4)
    assert np.bincount(first["split"]).tolist() == [12, 4, 4]
    assert first["metadata"]["cube_initial_position"] == [1.2, 1.2, 0.12]
    simulation = ArmSimulation(frame_skip=50)
    simulation.data.qpos[simulation.qpos_indices] = first["states"][0, 0, :2]
    observation = simulation.step(first["actions"][0, 0])
    assert observation["time"] == pytest.approx(DT)
    actual = np.concatenate([observation["joint_positions"], observation["joint_velocities"]])
    np.testing.assert_allclose(actual, first["states"][0, 1], atol=1e-10)
    inputs, deltas = transitions(first, 0)
    assert inputs.shape == (120, 6)
    assert deltas.shape == (120, 4)
    np.testing.assert_allclose(deltas[0], (first["states"][first["split"] == 0, 1] - first["states"][first["split"] == 0, 0])[0], atol=1e-6)


def test_baselines_and_physical_error_units():
    state = np.array([[0.1, 0.2, 1.0, -2.0]])
    np.testing.assert_array_equal(baseline(state, None), state)
    np.testing.assert_allclose(baseline(state, None, True), [[0.2, 0.0, 1.0, -2.0]])
    metrics = errors(state, np.zeros_like(state))
    np.testing.assert_allclose(metrics["angle_rmse_deg"], np.rad2deg([0.1, 0.2]))
    np.testing.assert_allclose(metrics["velocity_rmse_rad_s"], [1, 2])
    mean, scale = normalization(torch.ones(10, 6))
    assert torch.isfinite((torch.ones(10, 6) - mean) / scale).all()


def test_rollout_and_trajectory_use_previous_predictions(tmp_path, monkeypatch):
    from matplotlib.figure import Figure

    states = np.arange(2 * 13 * 4, dtype=float).reshape(2, 13, 4) * 100
    actions = np.arange(2 * 12 * 2, dtype=float).reshape(2, 12, 2) / 8
    dataset = {
        "states": states, "actions": actions,
        "split": np.array([2, 2]), "scenarios": np.array([0, 1]),
    }
    calls = []
    plotted = {}

    def increment(action):
        return np.concatenate([action, action + 1], axis=-1)

    def predictor(current, action):
        calls.append((current.copy(), action.copy()))
        return current + increment(action)

    def capture_plot(figure, path, **kwargs):
        if path.name == "trajectory.png":
            plotted["learned"] = [axes.lines[1].get_ydata().copy() for axes in figure.axes]

    monkeypatch.setattr(Figure, "savefig", capture_plot)
    report = evaluate(dataset, predictor, tmp_path)
    np.testing.assert_array_equal(calls[0][0], states[:, :-1])
    call_index = 1
    for horizon in range(1, 11):
        window_count = states.shape[1] - horizon
        expected = states[:, :window_count].copy()
        for offset in range(horizon):
            actual_input, actual_action = calls[call_index]
            expected_action = actions[:, offset:offset + window_count]
            np.testing.assert_array_equal(actual_input, expected)
            np.testing.assert_array_equal(actual_action, expected_action)
            expected += increment(expected_action)
            call_index += 1
        expected_error = errors(expected, states[:, horizon:])
        assert report["rollout"]["learned"][horizon - 1] == expected_error

    expected_trajectory = np.vstack([
        states[0, 0], states[0, 0] + np.cumsum(increment(actions[0, :10]), axis=0),
    ])
    for offset in range(10):
        actual_input, actual_action = calls[call_index]
        np.testing.assert_array_equal(actual_input, expected_trajectory[offset])
        np.testing.assert_array_equal(actual_action, actions[0, offset])
        call_index += 1
    assert call_index == len(calls)
    for joint in range(2):
        np.testing.assert_allclose(plotted["learned"][joint], np.rad2deg(expected_trajectory[:, joint]))


def test_training_roundtrip_and_evaluation(tmp_path):
    torch.set_num_threads(2)
    dataset = collect(tmp_path / "dataset.npz", episodes=20, steps=10)
    predictor = train(dataset, tmp_path / "run", epochs=10)
    inputs, deltas = transitions(dataset, 0)
    checkpoint = torch.load(tmp_path / "run/model.pt", weights_only=True)
    torch.testing.assert_close(checkpoint["input_mean"], inputs.mean(0))
    torch.testing.assert_close(checkpoint["delta_mean"], deltas.mean(0))
    current = dataset["states"][0, 0]
    action = dataset["actions"][0, 0]
    reloaded = load_predictor(tmp_path / "run/model.pt")
    np.testing.assert_array_equal(predictor(current, action), reloaded(current, action))
    report = evaluate(dataset, predictor, tmp_path / "run")
    assert len(report["rollout"]["learned"]) == 10
    for name in ("loss.png", "trajectory.png", "rollout_error.png", "evaluation.json"):
        assert (tmp_path / "run" / name).stat().st_size > 0


def test_model_can_overfit_small_physical_batch(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    dataset = collect(tmp_path / "tiny.npz", episodes=20, steps=10)
    inputs, deltas = transitions(dataset, 0)
    inputs, deltas = inputs[:16], deltas[:16]
    mean, scale = normalization(inputs)
    inputs = (inputs - mean) / scale
    mean, scale = normalization(deltas)
    deltas = (deltas - mean) / scale
    model = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    initial = torch.nn.functional.mse_loss(model(inputs), deltas).item()
    for _ in range(300):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(inputs), deltas)
        loss.backward()
        optimizer.step()
    assert torch.nn.functional.mse_loss(model(inputs), deltas).item() < initial * 0.01
