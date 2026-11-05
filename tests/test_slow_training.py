import numpy as np
import pytest

from simple_arm.slow_collection import _group, collect, commands
from simple_arm.slow_control_evaluation import compare
from simple_arm.slow_training import replay_mix, run


LIMITS = np.array([[-1.57, 1.57], [-2.1, 2.1]])


def test_slow_commands_include_initial_rate_bound_reversals_and_legal_targets():
    for seed in range(12):
        initial = np.array([1.31, -1.84]) if seed % 2 else np.array([-.4, .7])
        a = commands(np.random.default_rng(seed), initial, LIMITS, .2)
        b = commands(np.random.default_rng(seed), initial, LIMITS, .2)
        np.testing.assert_array_equal(a, b)
        assert a.shape == (40, 2)
        assert np.abs(np.diff(np.vstack([initial, a]), axis=0)).max() <= .02 + 1e-12
        assert np.all(a >= LIMITS[:, 0] + .15) and np.all(a <= LIMITS[:, 1] - .15)
        assert np.linalg.norm(a[0] - initial) < .003
    for rate in (0, -.2, float("nan")):
        with pytest.raises(ValueError):
            commands(np.random.default_rng(1), np.zeros(2), LIMITS, rate)


def dataset(seed, counts):
    n = sum(counts)
    states = np.zeros((n, 201, 13))
    states[..., 8] = 1
    return dict(states=states, actions=np.zeros((n, 200, 2)), contacts=np.zeros((n, 200, 10), dtype=bool),
                starting_contact=np.zeros((n, 200), dtype=bool), split=np.repeat([0, 1, 2], counts),
                kinds=np.tile(np.arange(4), n // 4), group_ids=np.arange(n) // 4,
                metadata={"seed": seed, "episodes": n, "command_rate_limit_rad_s": .2, "model_hash": "fixture"})


def test_replay_split_isolation_group_isolation_balancing_and_seed():
    slow, old = dataset(394, [8, 4, 4]), dataset(291, [16, 8, 4])
    merged = replay_mix(slow, old)
    assert np.bincount(merged["split"]).tolist() == [16, 8, 4]
    assert set(merged["metadata"]["parents"][i]["seed"] for i in range(2)) == {291, 394}
    for split in (0, 1):
        assert np.bincount(merged["source"][merged["split"] == split]).tolist() == ([8, 8] if split == 0 else [4, 4])
    for source, original in enumerate((slow, old)):
        mask = merged["source"] == source
        np.testing.assert_array_equal(merged["split"][mask], original["split"][merged["source_episode"][mask]])
    assert not ((merged["source"] == 1) & (merged["split"] == 2)).any()
    for group in np.unique(merged["group_ids"]):
        assert len(np.unique(merged["split"][merged["group_ids"] == group])) == 1
    np.testing.assert_array_equal(merged["source_episode"], replay_mix(slow, old)["source_episode"])


def test_small_collected_group_stationary_starts_and_actual_contacts():
    group, accepted, action, spec = _group((394, 0, .2))
    assert group == 0 and set(accepted) == {0, 1, 2, 3}
    assert spec["command_peak_rate_rad_s"] <= .2 + 1e-12
    control = action[::5]
    assert np.abs(np.diff(np.vstack([spec["initial_command"], control]), axis=0)).max() <= .02 + 1e-12
    for kind, (states, contacts, starting, peak) in accepted.items():
        assert states.shape == (201, 13) and contacts.shape == (200, 10)
        assert np.linalg.norm(states[0, 2:4]) == 0
        assert np.linalg.norm(states[0, 9:11]) == 0
        if kind <= 1:
            assert not contacts.any()
            assert np.linalg.norm(states[:, 4:6] - states[:1, 4:6], axis=1).max() <= .002
        else:
            assert contacts.any()


def test_overwrite_guards_and_collection_validation(tmp_path):
    path = tmp_path / "existing.npz"
    path.write_bytes(b"preserve")
    with pytest.raises(FileExistsError):
        collect(path)
    with pytest.raises(FileExistsError):
        run(tmp_path)
    with pytest.raises(FileExistsError):
        compare("missing.pt", tmp_path)
    with pytest.raises(ValueError):
        collect(tmp_path / "new.npz", rate=0)
    assert path.read_bytes() == b"preserve"
