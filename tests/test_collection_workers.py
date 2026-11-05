import multiprocessing as mp

import numpy as np
import pytest

from simple_arm.collection_workers import collect_tasks, effective_workers, unit_generator
from simple_arm.contact_boundary import _collect_group, classify, collect, load_boundary
from simple_arm.contact_varied import _collect_varied_episode


def test_worker_validation_and_independent_random_streams():
    for workers in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            effective_workers(workers, 10)
    assert effective_workers(8, 2) == 2
    a = unit_generator(7, 3, 10).random(20)
    np.testing.assert_array_equal(a, unit_generator(7, 3, 10).random(20))
    assert not np.array_equal(a, unit_generator(7, 4, 10).random(20))
    assert not np.array_equal(a, unit_generator(7, 3, 20).random(20))
    assert list(collect_tasks(_collect_group, [], workers=8)) == []


def test_worker_failure_propagates_and_pool_exits():
    children_before = {p.pid for p in mp.active_children()}
    with pytest.raises(ValueError):
        list(collect_tasks(_collect_varied_episode, [(-1, 0), (-1, 1)], workers=2))
    assert {p.pid for p in mp.active_children()} == children_before


def test_complete_boundary_collection_identical_with_one_and_eight_workers(tmp_path):
    serial_path, parallel_path = tmp_path / "serial.npz", tmp_path / "parallel.npz"
    serial = collect(serial_path, groups=40, seed=290, workers=1)
    parallel = collect(parallel_path, groups=40, seed=290, workers=8)
    assert serial["metadata"] == parallel["metadata"]
    for key in ("states", "actions", "contacts", "starting_contact", "split", "kinds", "group_ids", "diagnostics", "family"):
        np.testing.assert_array_equal(serial[key], parallel[key])
    loaded = load_boundary(parallel_path)
    assert loaded["metadata"] == serial["metadata"]
    for group in range(40):
        indices = np.flatnonzero(parallel["group_ids"] == group)
        assert len(np.unique(parallel["split"][indices])) == 1
        commands = parallel["actions"][indices]
        np.testing.assert_array_equal(commands, np.broadcast_to(commands[:1], commands.shape))
    for category, diagnostic in zip(parallel["kinds"], parallel["diagnostics"]):
        assert classify(diagnostic)[category]
    with pytest.raises(FileExistsError):
        collect(parallel_path, groups=40, workers=8)
