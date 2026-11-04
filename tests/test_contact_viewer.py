import mujoco
import numpy as np
import pytest

from simple_arm.contact_simulation import ContactSimulation
from simple_arm.contact_viewer import (
    choose_episode, draw_prediction, eligible_episodes, generate_random_episode,
    sample_state, set_display_state,
)


def test_replay_state_mapping_and_nonphysical_ghost():
    simulation = ContactSimulation()
    state = simulation.initialize([0.2, -0.3], [0.5, 0.2], 0.7)
    state[2:4] = [0.1, -0.2]
    state[9:13] = [0.03, -0.04, 0.01, 0.6]
    set_display_state(simulation, simulation.data, state)
    np.testing.assert_allclose(simulation.extract(), state, atol=1e-12)
    original = simulation.data.qpos.copy()
    ghost = mujoco.MjData(simulation.model)
    predicted = state.copy()
    predicted[4] += 0.1
    set_display_state(simulation, ghost, predicted)
    scene = mujoco.MjvScene(simulation.model, maxgeom=10)
    draw_prediction(scene, simulation, ghost)
    assert scene.ngeom == 4
    np.testing.assert_allclose(scene.geoms[3].pos, predicted[4:7], atol=1e-7)
    np.testing.assert_array_equal(simulation.data.qpos, original)


def test_display_interpolation_wraps_yaw_and_selects_episode():
    states = np.zeros((2, 13))
    states[:, 7] = np.sin(np.deg2rad([179, -179]))
    states[:, 8] = np.cos(np.deg2rad([179, -179]))
    midpoint = sample_state(states, 0.05, 0.1)
    np.testing.assert_allclose(midpoint[7:9], [0, -1], atol=1e-12)
    np.testing.assert_allclose(sample_state(states, 0.1, 0.1), states[-1], atol=1e-12)
    actual = np.stack([states.copy(), states.copy()])
    actual[:, -1, 4] = 0.1
    predicted = actual.copy()
    predicted[1, -1, 4] += 0.3
    assert choose_episode(actual, predicted, "best") == 0
    assert choose_episode(actual, predicted, "worst") == 1


def test_still_selection_checks_whole_path_not_only_endpoint():
    actual = np.zeros((3, 3, 13))
    predicted = actual.copy()
    actual[0, 1, 4] = 0.1  # Goes away and comes back; not a still episode.
    predicted[0, 1, 4] = 2
    predicted[1, :, 4] = 0.01
    predicted[2, 1:, 4] = 0.1
    np.testing.assert_array_equal(eligible_episodes(actual, "still"), [1, 2])
    assert choose_episode(actual, predicted, "worst-still") == 2
    with pytest.raises(ValueError, match="No episodes"):
        eligible_episodes(actual[:1], "still")


def test_random_still_physics_reproducibility_and_recursive_feedback():
    calls = []

    def predictor(current, action):
        calls.append(current.copy())
        result = current.copy()
        result[..., 4] += 0.001
        return result

    actual, predicted, info = generate_random_episode(2026, predictor, condition="still")
    assert actual.shape == predicted.shape == (41, 13)
    assert info["actual_max_motion_mm"] <= 2
    np.testing.assert_array_equal(np.asarray(calls)[:, 0], predicted[:-1])
    np.testing.assert_allclose(predicted[-1, 4] - predicted[0, 4], 0.04)
    repeated, _, repeated_info = generate_random_episode(2026, predictor, condition="still")
    np.testing.assert_array_equal(actual, repeated)
    assert repeated_info == info
    with pytest.raises(RuntimeError, match="after 0 attempts"):
        generate_random_episode(1, predictor, max_attempts=0)
