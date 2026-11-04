import hashlib

import numpy as np
import pytest
import torch

from simple_arm.arm_learning import DT, STATE_NAMES, make_model
from simple_arm.mpc import Config, LearnedMPC, load_dynamics, metrics, rollout, run_episode, trajectory_cost
from simple_arm.simulation import MODEL_PATH


LIMITS = np.array([[-1.57, 1.57], [-2.1, 2.1]])


def simple_prediction(current, action):
    return np.concatenate([action, np.zeros_like(action)], axis=-1)


def test_checkpoint_normalization_and_compatibility(tmp_path):
    torch.manual_seed(10)
    model = make_model().eval()
    checkpoint = {
        "weights": model.state_dict(),
        "input_mean": torch.arange(6, dtype=torch.float32),
        "input_scale": torch.arange(1, 7, dtype=torch.float32),
        "delta_mean": torch.arange(4, dtype=torch.float32),
        "delta_scale": torch.arange(1, 5, dtype=torch.float32),
        "metadata": {"state_order": STATE_NAMES, "dt": DT, "model_sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()},
    }
    path = tmp_path / "model.pt"
    torch.save(checkpoint, path)
    before = path.read_bytes()
    predict = load_dynamics(path)
    current, action = np.array([[0.3, 0.4, 0.5, 0.6]]), np.array([[0.7, 0.8]])
    inputs = torch.tensor(np.concatenate([current, action], axis=-1), dtype=torch.float32)
    with torch.no_grad():
        expected = current + (model((inputs - checkpoint["input_mean"]) / checkpoint["input_scale"]) * checkpoint["delta_scale"] + checkpoint["delta_mean"]).numpy()
    np.testing.assert_allclose(predict(current, action), expected)
    assert path.read_bytes() == before
    checkpoint["input_scale"][0] = 0
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="positive"):
        load_dynamics(path)


def test_batched_rollout_is_recursive_and_unclipped():
    actions = np.ones((8, 5, 2))
    inputs = []

    def predict(current, action):
        inputs.append(current.copy())
        return current + np.tile(action, (1, 2))

    predicted = rollout(predict, np.zeros(4), actions)
    assert predicted.shape == (8, 5, 4)
    for step in range(5):
        np.testing.assert_array_equal(inputs[step], np.full((8, 4), step))
        np.testing.assert_array_equal(predicted[:, step], np.full((8, 4), step + 1))


def test_cost_includes_previous_command_and_terminal_terms():
    config = Config()
    predicted = np.array([[[0.2, 0.1, 0.3, -0.4], [0.4, 0.2, 0.1, -0.2]]])
    actions = np.array([[[0.2, 0.3], [0.4, 0.5]]])
    previous = np.array([0.1, -0.1])
    expected = (0.05 + 0.2) + 0.03 * (0.25 + 0.05) + 0.08 * (0.01 + 0.16 + 0.04 + 0.04) + 8 * 0.2 + 0.1 * 0.05
    cost = trajectory_cost(predicted, actions, np.zeros(2), previous, LIMITS, config)
    assert cost[0] == pytest.approx(expected)
    predicted[0, 1, 0] = 2
    assert np.isinf(trajectory_cost(predicted, actions, np.zeros(2), previous, LIMITS, config)[0])


def test_actions_legal_deterministic_and_model_changes_selection():
    config = Config(horizon=3)

    def biased_prediction(current, action):
        return simple_prediction(current, action * 2)

    first = LearnedMPC(biased_prediction, LIMITS, LIMITS, config)
    second = LearnedMPC(biased_prediction, LIMITS, LIMITS, config)
    observed, goal, previous = np.zeros(4), np.array([0.6, -0.8]), np.zeros(2)
    command, info = first.act(observed, goal, previous)
    other_command, other_info = second.act(observed, goal, previous)
    np.testing.assert_array_equal(command, other_command)
    assert info["selected"] == other_info["selected"]
    assert info["selected"] != 0
    assert np.linalg.norm(2 * command - goal) < np.linalg.norm(goal)
    candidates = first.candidates(goal, previous)
    assert np.all(candidates >= first.bounds[:, 0]) and np.all(candidates <= first.bounds[:, 1])


def test_infeasible_rollouts_use_explicit_legal_fallback():
    def invalid_prediction(current, action):
        return np.full_like(current, 100)

    planner = LearnedMPC(invalid_prediction, LIMITS, LIMITS)
    previous = np.array([0.1, 0.2])
    command, diagnostic = planner.act(np.zeros(4), np.zeros(2), previous)
    np.testing.assert_array_equal(command, previous)
    assert diagnostic["fallback"]
    assert diagnostic["rejected"] == 256


def test_replanning_reads_actual_state_and_previous_executed_command(monkeypatch):
    calls = []

    def act(self, observed, goal, previous):
        calls.append((observed.copy(), previous.copy()))
        command = goal * (0.5 + 0.1 * len(calls))
        return command, {"fallback": False, "selected": 3, "rejected": 0, "cost": 0.0}

    monkeypatch.setattr(LearnedMPC, "act", act)
    scenario = {"initial": [0, 0], "duration": 0.3, "targets": [(0, [0.3, -0.4])]}
    episode = run_episode(scenario, "mpc", simple_prediction, Config())
    assert len(calls) == 3
    assert len(episode["fine_states"]) == 150
    for index, (observed, previous) in enumerate(calls):
        np.testing.assert_array_equal(observed, episode["states"][index])
        np.testing.assert_array_equal(previous, episode["commands"][index - 1] if index else [0, 0])


def test_episode_reproducibility_and_metrics():
    scenario = {"initial": [0, 0], "duration": 0.4, "targets": [(0, [0.2, -0.3])]}
    first = run_episode(scenario, "mpc", simple_prediction, Config(horizon=3, candidates=32))
    second = run_episode(scenario, "mpc", simple_prediction, Config(horizon=3, candidates=32))
    np.testing.assert_array_equal(first["commands"], second["commands"])
    np.testing.assert_array_equal(first["fine_states"], second["fine_states"])
    summary = metrics(first, scenario, LIMITS)
    expected = np.rad2deg(np.sqrt(np.mean((first["fine_states"][:, :2] - [0.2, -0.3]) ** 2, axis=0)))
    np.testing.assert_allclose(summary["tracking_rmse_deg"], expected)
    assert summary["limit_violation_samples"] == 0
