import numpy as np
import pytest

from simple_arm.collection_workers import collect_tasks
from simple_arm.contact_boundary import _collect_group, classify
from simple_arm.contact_simulation import ContactSimulation
from simple_arm.contact_varied import _collect_varied_episode, limited_commands
from simple_arm.rate_collection import check_rate


def test_rate_limiter_includes_first_command_and_all_families():
    limits = ContactSimulation().action_limits
    rng = np.random.default_rng(12)
    for family in range(4):
        for _ in range(30):
            initial = rng.uniform(limits[:, 0] + .2, limits[:, 1] - .2)
            commands = limited_commands(rng, initial, limits, family, .2)
            assert commands.shape == (40, 2)
            assert np.abs(np.diff(np.vstack([initial, commands]), axis=0)).max() <= .02 + 1e-12
            assert (commands >= limits[:, 0] + .15).all()
            assert (commands <= limits[:, 1] - .15).all()
    for rate in (0, -1, float("nan")):
        with pytest.raises(ValueError):
            limited_commands(rng, initial, limits, 0, rate)


def test_rate_limited_physics_and_group_categories():
    _, found, actions, _ = _collect_group((590, 0, .2))
    initial = found[0][0][0, :2]
    assert np.abs(np.diff(np.vstack([initial, actions[::5]]), axis=0)).max() <= .02 + 1e-12
    for category, (states, contacts, _, diagnostic) in found.items():
        assert classify(diagnostic)[category]
        np.testing.assert_array_equal(states[0, :2], initial)
        np.testing.assert_array_equal(states[0, 2:4], [0, 0])
        assert states.shape == (201, 13) and contacts.shape == (200, 10)


def test_limited_collection_independent_of_worker_scheduling():
    task = (491, 0, .2)
    a = _collect_varied_episode(task)
    b = list(collect_tasks(_collect_varied_episode, [task], workers=2))[0]
    for first, second in zip(a[1:5], b[1:5]):
        np.testing.assert_array_equal(first, second)
    states, actions, contacts = a[1:4]
    stats = check_rate({"states": states[None], "actions": actions[None],
                        "contacts": contacts[None], "split": np.array([0])})
    assert stats["peak_command_rate_rad_s"] <= .2 + 1e-12
    invalid = actions.copy()
    invalid[0] += 1
    with pytest.raises(ValueError, match="slew"):
        check_rate({"states": states[None], "actions": invalid[None]})


def test_merge_reports_peak_over_both_sources(tmp_path):
    from simple_arm.contact_boundary import merge
    from test_residual_training import numerical_data, save_fixture
    broad, focused = numerical_data(), numerical_data()
    focused["diagnostics"] = np.zeros((3, 6))
    focused["family"] = np.zeros(3, dtype=int)
    for data, speed in ((broad, .4), (focused, .25)):
        data["states"][..., 2:4] = speed
        data["metadata"].update(command_rate_limit_rad_s=.2, measured_peak_command_rate_rad_s=.2,
                                measured_peak_actual_joint_speed_rad_s=speed)
    save_fixture(tmp_path / "broad.npz", broad)
    save_fixture(tmp_path / "focused.npz", focused)
    combined = merge(tmp_path / "focused.npz", tmp_path / "broad.npz", tmp_path / "mixed.npz")
    assert combined["metadata"]["measured_peak_actual_joint_speed_rad_s"] == .4
