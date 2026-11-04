import json

import numpy as np
import pytest

from simple_arm.contact_boundary import (
    boundary_offset, classify, compare_results, evaluate_boundary, fingerprint, group_split, load_boundary, merge,
    pusher_gap, save_dataset, simulate,
)
from simple_arm.contact_learning import features, training_arrays
from simple_arm.contact_simulation import ContactSimulation, link_lengths, model_hash


def test_boundary_categories_and_group_split():
    fixtures = [[0, 0, .002, .1, .01, 1], [0, 0, .1, .1, .002, 0],
                [.01, .1, .1, .1, -.001, 0], [.06, .15, .1, .1, -.002, 0]]
    np.testing.assert_array_equal([classify(row) for row in fixtures], np.eye(4, dtype=bool))
    assert not classify([.003, 0, .002, .1, .01, 1]).any()
    assert not classify([.01, .3, .1, .1, -.001, 0]).any()
    first = group_split(40, np.random.default_rng(5))
    np.testing.assert_array_equal(first, group_split(40, np.random.default_rng(5)))
    np.testing.assert_array_equal(np.bincount(first), [112, 24, 24])
    assert np.all(first.reshape(-1, 4) == first.reshape(-1, 4)[:, :1])
    with pytest.raises(ValueError):
        group_split(3, np.random.default_rng(5))


def test_clearance_bracketing_and_dense_ground_truth_diagnostics():
    path = np.stack([np.linspace(.3, .5, 100), np.zeros(100)], axis=-1)
    normal, point = np.array([0, 1]), np.array([.4, 0])
    offset = boundary_offset(path, point, normal, 0, .002)
    assert pusher_gap(path, point + offset * normal, 0) == pytest.approx(.002, abs=1e-7)
    simulation = ContactSimulation()
    result = simulate(simulation, [0, 0], [.2, 0], 0, [0, 0], np.zeros((40, 2)))
    assert result[0].shape == (201, 13)
    assert result[1].shape == (200, 10)
    assert result[4].shape == (2001, 2)
    assert classify(result[3])[0]  # Link directly over cube, pusher safely away.
    repeated = simulate(simulation, [0, 0], [.2, 0], 0, [0, 0], np.zeros((40, 2)))
    for first, second in zip(result, repeated):
        np.testing.assert_array_equal(first, second)


def test_merge_preserves_original_splits_and_diagnostics_are_not_inputs(tmp_path):
    simulation = ContactSimulation()
    result = simulate(simulation, [0, 0], [.2, 0], 0, [0, 0], np.zeros((40, 2)))
    n = 12
    data = dict(states=np.repeat(result[0][None], n, axis=0), actions=np.zeros((n, 200, 2)),
                contacts=np.repeat(result[1][None], n, axis=0), starting_contact=np.repeat(result[2][None], n, axis=0),
                split=np.repeat([0, 1, 2], 4), kinds=np.tile(np.arange(4), 3), group_ids=np.repeat(np.arange(3), 4),
                diagnostics=np.repeat(result[3][None], n, axis=0), family=np.zeros(n, dtype=int),
                metadata={"seed": 1, "lengths": link_lengths(), "model_hash": model_hash(), "kinds": ["a", "b", "c", "d"]})
    first = tmp_path / "focused.npz"
    broad = tmp_path / "broad.npz"
    save_dataset(first, data, {})
    save_dataset(broad, data, {})
    merged = merge(first, broad, tmp_path / "mixed.npz")
    loaded = load_boundary(tmp_path / "mixed.npz")
    np.testing.assert_array_equal(loaded["split"][:n], data["split"])
    np.testing.assert_array_equal(loaded["split"][n:], data["split"])
    assert len(np.unique(loaded["group_ids"][:n])) == n
    assert fingerprint(loaded) == merged["metadata"]["data_sha256"]
    inputs, targets, weights = training_arrays(loaded, 5, 0)
    changed = dict(loaded, diagnostics=np.ones_like(loaded["diagnostics"]) * 99)
    other_inputs, other_targets, _ = training_arrays(changed, 5, 0)
    np.testing.assert_array_equal(inputs, other_inputs)
    np.testing.assert_array_equal(targets, other_targets)
    np.testing.assert_allclose(inputs.numpy().mean(axis=0), features(loaded["states"][loaded["split"] == 0, :-1:5], loaded["actions"][loaded["split"] == 0, ::5], link_lengths()).reshape(-1, 15).mean(axis=0), atol=1e-6)
    with pytest.raises(FileExistsError):
        save_dataset(first, data, {})
    corrupted = dict(data, split=data["split"].copy(), metadata=dict(data["metadata"]))
    corrupted["split"][0] = 1
    save_dataset(tmp_path / "corrupted.npz", corrupted, {})
    with pytest.raises(ValueError, match="crosses"):
        load_boundary(tmp_path / "corrupted.npz")


def test_boundary_evaluation_rejects_parent_seed_overlap(tmp_path, monkeypatch):
    saved = {"metadata": {"seed": 291, "parents": [{"seed": 191, "data_sha256": "original"}]}}
    monkeypatch.setattr("simple_arm.contact_boundary.torch.load", lambda *args, **kwargs: saved)
    dataset = {"metadata": {"seed": 191, "data_sha256": "changed"}}
    with pytest.raises(ValueError, match="overlaps"):
        evaluate_boundary(dataset, "unused.pt", tmp_path / "results", independent=True)
    with pytest.raises(ValueError, match="provenance"):
        evaluate_boundary(dataset, "unused.pt", tmp_path / "results")


def test_comparison_requires_drift_and_pushing_targets(tmp_path):
    for name, drift, push in (("old", .03, .1), ("firstpass", .025, .12), ("new", .01, .105)):
        for prefix in ("boundary", "broad"):
            directory = tmp_path / f"{prefix}_{name}"
            directory.mkdir()
            report = {"category_metrics": {k: {"learned": {"mean_max_motion_m": drift}} for k in ("overhead", "near_miss")},
                      "endpoint": {"learned": {"displacement_vector_rmse_m": push}},
                      "moving_episode_endpoint": {"learned": {"displacement_vector_rmse_m": push}}}
            (directory / "evaluation.json").write_text(json.dumps(report))
    result = compare_results(tmp_path)
    assert not result["models"]["firstpass"]["passes_drift_target"]
    assert not result["models"]["firstpass"]["passes_pushing_target"]
    assert result["models"]["refined"]["passes_drift_target"]
    assert result["models"]["refined"]["passes_pushing_target"]
