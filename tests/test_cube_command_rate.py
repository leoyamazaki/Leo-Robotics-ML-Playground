from dataclasses import replace

import mujoco
import numpy as np
import pytest
import torch

from simple_arm.cube_mpc import CEMController, Config, DT, limit_command_rate, validate_command_rate
from simple_arm.cube_mpc_experiment import initialize, load_cases, run_episode
from simple_arm.cube_mpc import DATASET
from simple_arm.cube_rate_experiment import compare, motion_metrics


LIMITS = np.array([[-1.57, 1.57], [-2.1, 2.1]])


def dynamics(state, action):
    result = state.clone()
    result[..., 4:6] += action * .02
    return result


def assert_rate(actions, previous, limit):
    changes = np.diff(np.vstack([previous, actions]), axis=0)
    assert np.abs(changes).max() <= limit * DT + 1e-6


def test_sequential_projection_both_joints_first_step_and_disabled_identity():
    requested = torch.tensor([[[2., -2.], [2., -2.], [-2., 2.]], [[-.5, .5], [0., 0.], [1., -1.]]])
    previous = torch.tensor([.1, -.1])
    limited = limit_command_rate(requested, previous, .5)
    torch.testing.assert_close(limited[0], torch.tensor([[.15, -.15], [.2, -.2], [.15, -.15]]))
    for actions in limited:
        assert_rate(actions.numpy(), previous.numpy(), .5)
    assert limit_command_rate(requested, previous, 0.) is requested
    torch.testing.assert_close(limit_command_rate(torch.zeros(2, 4, 2), torch.zeros(2), .5), torch.zeros(2, 4, 2))


def test_every_neural_candidate_is_limited_and_winning_prediction_matches(monkeypatch):
    monkeypatch.setattr(mujoco, "mj_step", lambda *a, **kw: pytest.fail("Physics in planner"))
    calls = []
    def spy(state, action):
        calls.append(action.clone())
        return dynamics(state, action)
    c = Config(horizon=6, knots=3, candidates=20, iterations=3, command_rate_limit=.5)
    planner = CEMController(spy, LIMITS, LIMITS, c)
    observed = np.zeros(13)
    previous = np.array([.2, -.2])
    result = planner.plan(observed, [.1, 0], previous)
    for start in range(0, len(calls), c.horizon):
        actions = torch.stack(calls[start:start + c.horizon], 1).numpy()
        for candidate in actions:
            assert_rate(candidate, previous, .5)
    assert_rate(result.actions, previous, .5)
    np.testing.assert_allclose(result.predicted[:, 4:6], np.cumsum(.02 * result.actions, axis=0), atol=1e-7)
    assert result.cost == pytest.approx(sum(result.diagnostics["score_components"].values()), rel=1e-6)
    assert result.diagnostics["selected_max_command_rate_rad_s"] <= .5 + 1e-5
    assert np.all(result.actions >= LIMITS[:, 0] + .15) and np.all(result.actions <= LIMITS[:, 1] - .15)


def test_warm_start_reset_and_fallback_preserve_rate():
    c = Config(horizon=6, knots=3, candidates=20, iterations=2, command_rate_limit=.5)
    planner = CEMController(dynamics, LIMITS, LIMITS, c)
    observed, previous = np.zeros(13), np.array([1.4, -1.9])
    first = planner.plan(observed, [.3, -.1], previous)
    second = planner.plan(observed, [-.3, .1], first.actions[0])
    assert_rate(first.actions, previous, .5)
    assert_rate(second.actions, first.actions[0], .5)
    planner.reset()
    np.testing.assert_array_equal(first.actions, planner.plan(observed, [.3, -.1], previous).actions)
    # Reset on a changed target still starts at the actual previous command.
    planner.reset()
    reset = planner.plan(observed, [0., 0.], second.actions[0])
    assert_rate(reset.actions, second.actions[0], .5)
    def invalid(s, a):
        return torch.full_like(s, torch.nan)
    fallback = CEMController(invalid, LIMITS, LIMITS, c).plan(observed, [0, 0], previous)
    assert fallback.diagnostics["fallback"]
    np.testing.assert_allclose(fallback.actions, np.tile(previous, (6, 1)))
    assert_rate(fallback.actions, previous, .5)


def test_execution_guard_does_not_silently_change_command():
    validate_command_rate([.05, -.05], np.zeros(2), .5)
    with pytest.raises(RuntimeError, match="rate"):
        validate_command_rate([.06, 0], np.zeros(2), .5)
    validate_command_rate([1., -1.], np.zeros(2), 0.)
    for rate in (-.1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            Config(command_rate_limit=rate)


def test_actual_execution_interval_initialization_and_motion_logging():
    initial, target = load_cases(DATASET)[23]
    c = Config(horizon=3, knots=2, candidates=8, iterations=1, command_rate_limit=.5)
    e = run_episode(initial, target, dynamics, c, duration=.3)
    assert e["fine_states"].shape == (150, 13)
    np.testing.assert_allclose(e["states"][0], initialize(initial).extract())
    assert_rate(e["commands"], initial[:2], .5)
    report = motion_metrics(e)
    assert report["maximum_command_rate_rad_s"] <= .5 + 1e-5
    assert report["rms_command_rate_rad_s"] <= .5 + 1e-5
    assert np.isfinite(report["maximum_actual_joint_speed_rad_s"])


def test_disabled_limit_preserves_seeded_sequences(tmp_path):
    c = Config(horizon=6, knots=3, candidates=20, iterations=2, command_rate_limit=0.)
    a, b = [CEMController(dynamics, LIMITS, LIMITS, settings) for settings in (c, replace(c, command_rate_limit=0.))]
    for _ in range(2):
        first, second = [p.plan(np.zeros(13), [.1, .05], [0, 0]) for p in (a, b)]
        np.testing.assert_array_equal(first.actions, second.actions)
        assert first.cost == second.cost
    with pytest.raises(FileExistsError):
        compare("missing.pt", "missing.npz", tmp_path, c)


def test_shared_default_is_point_two_and_zero_disables():
    assert Config().command_rate_limit == .2
    requested = torch.ones(1, 2, 2)
    limited = limit_command_rate(requested, torch.zeros(2), Config().command_rate_limit)
    torch.testing.assert_close(limited[0, :, 0], torch.tensor([.02, .04]))
    assert limit_command_rate(requested, torch.zeros(2), 0.) is requested
