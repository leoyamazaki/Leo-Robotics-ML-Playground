import numpy as np

from simple_arm.rollout_stress import autoregressive, make_cases, summarize


def test_stress_rollout_never_receives_future_truth():
    initial = np.array([1.0, 2.0, 3.0, 4.0])
    actions = np.arange(100).reshape(50, 2) / 8
    calls = []

    def predict(current, action):
        calls.append(current.copy())
        return current + np.tile(action, 2)

    trajectory = autoregressive(predict, initial, actions)
    expected = initial + np.cumsum(np.tile(actions, (1, 2)), axis=0)
    np.testing.assert_array_equal(trajectory[1:], expected)
    np.testing.assert_array_equal(calls, trajectory[:-1])
    np.testing.assert_array_equal(initial, [1, 2, 3, 4])


def test_stress_cases_and_limit_metrics():
    limits = np.array([[-1.57, 1.57], [-2.1, 2.1]])
    cases = make_cases(limits, count=2)
    assert len(cases) == 8
    for group, initial, actions in cases:
        assert actions.shape == (50, 2)
        assert np.all(actions >= limits[:, 0]) and np.all(actions <= limits[:, 1])
        if group.startswith("limit"):
            assert np.all(np.abs(initial) >= 0.94 * limits[:, 1])
    actual = np.zeros((1, 51, 4))
    predicted = actual.copy()
    predicted[:, 1:, 0] = 2
    actions = np.tile([1.0, 1.0], (1, 50, 1))
    actions[:, 25:] *= -1
    result = summarize(predicted, actual, actions, limits)
    assert result["predicted_limit_exceedance_samples"] == 50
    assert result["actual_limit_exceedance_samples"] == 0
    assert result["first_step_after_reversal"]["samples"] == 1
    assert result["near_limit"]["samples"] == 0
