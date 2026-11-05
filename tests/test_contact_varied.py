import numpy as np
import pytest

from simple_arm.contact_simulation import ContactSimulation
from simple_arm.contact_varied import check_coverage, collect_varied, command_sequence, evaluate_varied, summarize
from simple_arm.contact_learning import load_dataset, training_arrays


def test_varied_initial_velocity_and_validation():
    simulation = ContactSimulation()
    state = simulation.initialize([0, 0], [0.6, 0.2], 1.2, [0.2, -0.3], [0.1, -0.05, 0.7])
    np.testing.assert_allclose(state[[2, 3, 9, 10, 12]], [0.2, -0.3, 0.1, -0.05, 0.7])
    with pytest.raises(ValueError, match="Cube velocity"):
        simulation.initialize([0, 0], [0.6, 0.2], 0, cube_velocity=[np.nan, 0, 0])
    with pytest.raises(ValueError, match="joint limits"):
        simulation.initialize([10, 0], [0.6, 0.2], 0)


def test_varied_command_families_are_legal_and_move_both_joints():
    simulation = ContactSimulation()
    for family in range(4):
        first = command_sequence(np.random.default_rng(94), np.zeros(2), simulation.action_limits, family)
        second = command_sequence(np.random.default_rng(94), np.zeros(2), simulation.action_limits, family)
        np.testing.assert_array_equal(first, second)
        assert np.all(first >= simulation.action_limits[:, 0] + 0.15)
        assert np.all(first <= simulation.action_limits[:, 1] - 0.15)
        assert np.all(np.ptp(first, axis=0) > np.deg2rad(15))


def test_varied_evaluation_rejects_potential_episode_overlap(tmp_path, monkeypatch):
    saved = {"stride": 5, "metadata": {"seed": 191, "data_sha256": "original"}}
    monkeypatch.setattr("simple_arm.contact_varied.torch.load", lambda *args, **kwargs: saved)
    dataset = {"metadata": {"seed": 191, "data_sha256": "different_episode_split"}}
    with pytest.raises(ValueError, match="provenance"):
        evaluate_varied(dataset, "unused.pt", tmp_path / "results")
    with pytest.raises(ValueError, match="different seed"):
        evaluate_varied(dataset, "unused.pt", tmp_path / "results", all_episodes=True)


def test_varied_collection_reproducible_and_coverage(tmp_path):
    path = tmp_path / "varied.npz"
    first = collect_varied(path, episodes=40, seed=191, workers=1)
    second = collect_varied(tmp_path / "repeated.npz", episodes=40, seed=191, workers=8)
    for key in ("states", "actions", "contacts", "starting_contact", "split", "kinds"):
        np.testing.assert_array_equal(first[key], second[key])
    assert first["metadata"] == second["metadata"]
    check_coverage(summarize(first))
    for split in range(3):
        assert set(first["kinds"][first["split"] == split]) == set(range(4))
    assert np.all(np.linalg.norm(first["states"][:, 0, 9:11], axis=-1)[4::8] > 0)
    assert np.count_nonzero(np.linalg.norm(first["states"][:, 0, 9:11], axis=-1) < 1e-12) == 20
    held = first["actions"].reshape(40, 40, 5, 2)
    np.testing.assert_array_equal(held, np.broadcast_to(held[:, :, :1], held.shape))
    loaded = load_dataset(path)
    assert loaded["metadata"] == first["metadata"]
    inputs, targets, _ = training_arrays(loaded, 5, 0)
    assert inputs.shape[1] == 15 and targets.shape[1] == 12
    with pytest.raises(FileExistsError):
        collect_varied(path, episodes=40)
    bad_summary = dict(summarize(first), contact_window_fraction=0)
    with pytest.raises(RuntimeError, match="coverage"):
        check_coverage(bad_summary)
