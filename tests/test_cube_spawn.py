import numpy as np
import pytest
import torch

from simple_arm.contact_varied import tip_position
from simple_arm.cube_mpc import DATASET
from simple_arm.cube_mpc_experiment import initialize, load_cases
from simple_arm.cube_mpc_viewer import spawn_scenario
from simple_arm.cube_workspace import Workspace


def test_default_spawn_exactly_preserves_saved_scenario():
    for initial, target in load_cases(DATASET).values():
        actual, goal = spawn_scenario(initial, target)
        np.testing.assert_array_equal(actual, initial)
        np.testing.assert_array_equal(goal, target)


def test_random_spawn_reproducible_near_arm_and_reachable_without_contact():
    w = Workspace()
    for base, goal in load_cases(DATASET).values():
        original = base.copy()
        for seed in (0, 1, 42, 43):
            initial, target = spawn_scenario(base, goal, spawn_seed=seed)
            again, again_target = spawn_scenario(base, goal, spawn_seed=seed)
            np.testing.assert_array_equal(initial, again)
            np.testing.assert_array_equal(target, again_target)
            tip = tip_position(base[:2], w.lengths)
            assert .08 < np.linalg.norm(initial[4:6] - tip) <= .2
            assert np.linalg.norm(initial[4:6]) <= sum(w.lengths) - w.contact_radius
            assert w.tip_distance(torch.tensor(initial[4:6], dtype=torch.float32)).item() < 1e-6
            assert w.tip_distance(torch.tensor(target, dtype=torch.float32)).item() < 1e-6
            np.testing.assert_allclose(target - initial[4:6], goal - base[4:6])
            np.testing.assert_array_equal(initial[:4], base[:4])
            assert not initialize(initial).contact()
        np.testing.assert_array_equal(base, original)
        a = spawn_scenario(base, goal, spawn_seed=42)[0]
        b = spawn_scenario(base, goal, spawn_seed=43)[0]
        assert not np.array_equal(a[4:6], b[4:6])


def test_custom_spawn_translated_goal_and_explicit_target():
    base, demonstrated = load_cases(DATASET)[99]
    initial, target = spawn_scenario(base, demonstrated, cube=[.4, .05])
    np.testing.assert_array_equal(initial[4:6], [.4, .05])
    np.testing.assert_allclose(target - initial[4:6], demonstrated - base[4:6])
    _, target = spawn_scenario(base, demonstrated, target=[.35, .1], cube=[.4, .05])
    np.testing.assert_array_equal(target, [.35, .1])
    _, target = spawn_scenario(base, demonstrated, target=[.35, .1], spawn_seed=42)
    np.testing.assert_array_equal(target, [.35, .1])


def test_invalid_or_overlapping_spawn_rejected():
    base, target = load_cases(DATASET)[99]
    for options in ({"spawn_seed": -1}, {"cube": [float("nan"), 0]}, {"cube": [0]},
                    {"cube": [0, 0], "spawn_seed": 1}, {"target": [float("inf"), 0]}):
        with pytest.raises(ValueError):
            spawn_scenario(base, target, **options)
    with pytest.raises(ValueError, match="overlaps"):
        spawn_scenario(base, target, cube=tip_position(base[:2], Workspace().lengths))
    with pytest.raises(ValueError, match="overlaps"):
        spawn_scenario(base, target, cube=[0, 0])
