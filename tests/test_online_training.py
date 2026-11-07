"""Small real-simulator runs; no synthetic data substitutes for online training."""

from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from simple_arm.online_data import (EpisodeStream, ReplayBuffer, dataset_from_rows,
                                    random_commands, random_episode, save_archive)
from simple_arm.online_training import Config, online_loss, run, sample_batch
from simple_arm.contact_learning import load_predictor
from simple_arm.contact_simulation import ContactSimulation


def boundary_fixture(path):
    data = dataset_from_rows([random_episode((591, i)) for i in range(3)], 591)
    data["split"] = np.arange(3)
    save_archive(path, data)


def tiny_config():
    return Config(one_step_epochs=2, recursive_epochs=1, one_step_new=4, recursive_new=2,
                  one_step_updates=2, recursive_updates=1, capacity=8, batch_size=8, heldout_episodes=3)


def test_random_commands_and_unfiltered_deterministic_physics():
    sim = ContactSimulation()
    for family in range(6):
        commands = random_commands(np.random.default_rng(55), np.zeros(2), sim.action_limits, family)
        assert np.abs(np.diff(np.vstack([np.zeros(2), commands]), axis=0)).max() <= .02 + 1e-12
        if family == 0:
            assert not commands.any()
        if family == 1:
            assert not commands[:, 1].any()
        if family == 2:
            assert not commands[:, 0].any()
    a, b = random_episode((791, 7)), random_episode((791, 7))
    for x, y in zip(a[:4], b[:4]):
        np.testing.assert_array_equal(x, y)
    assert a[4] == 0 and not a[2].any()  # A no-contact idle episode is retained.
    with EpisodeStream(791, 7, 9, workers=2, prefetch=2) as stream:
        rows = stream.take(2)
        np.testing.assert_array_equal(rows[0][0], a[0])
        assert [r[5]["episode_id"] for r in rows] == [7, 8]


def test_complete_episode_fifo_and_recursive_windows():
    replay = ReplayBuffer(3)
    first = dataset_from_rows([random_episode((791, i)) for i in range(2)], 791)
    replay.append(first)
    replay.append(dataset_from_rows([random_episode((791, i)) for i in range(2, 4)], 791))
    np.testing.assert_array_equal(replay.data["episode_ids"], [1, 2, 3])
    from simple_arm.residual_training import episode_arrays
    arrays = episode_arrays(replay.data, 0)
    config = replace(tiny_config(), batch_size=64)
    values, boundary_count = sample_batch(arrays, arrays, config, torch.Generator().manual_seed(91), True, "cpu")
    states, actions, contacts = values
    assert states.shape == (64, 11, 13) and actions.shape == (64, 10, 2)
    assert contacts.shape == (64, 10) and 0 <= boundary_count <= 64
    # Every complete sampled sequence occurs inside one saved episode, never across boundaries.
    for row in states:
        assert any(torch.equal(row, episode[start:start + 11]) for episode in arrays[0] for start in range(31))


def test_losses_have_no_stationary_or_contact_multipliers():
    from simple_arm.contact_learning import make_model
    from simple_arm.residual_training import recursive_loss
    from test_residual_training import numerical_data, reference
    model = make_model(16)
    x, y = torch.randn(2, 15), torch.randn(2, 12)
    torch.testing.assert_close(online_loss(model, (x, y), {}, False), (model(x) - y).square().mean())
    data = numerical_data()
    c = reference(data, False)
    states = torch.tensor(data["states"][:1, :55:5], dtype=torch.float32)
    actions = torch.tensor(data["actions"][:1, :50:5], dtype=torch.float32)
    contacts = torch.ones(1, 10, dtype=torch.bool)
    uniform = online_loss(model, (states, actions, contacts), c, True)
    weighted = recursive_loss(model, states, actions, contacts, c)
    torch.testing.assert_close(weighted, uniform * 3)


def test_cpu_online_resume_matches_uninterrupted_and_status_is_quiet(tmp_path, capsys):
    torch.set_num_threads(2)
    boundary = tmp_path / "boundary.npz"
    boundary_fixture(boundary)
    config = tiny_config()
    full, partial, resumed = (tmp_path / name for name in ("full", "partial", "resumed"))
    run(full, config, "cpu", 1, boundary, evaluate=False)
    run(partial, replace(config, one_step_epochs=1, recursive_epochs=0), "cpu", 1, boundary, evaluate=False)
    run(resumed, config, "cpu", 2, boundary, partial / "one_step/last.pt", evaluate=False)
    for stage in ("one_step", "recursive"):
        a, b = [torch.load(root / stage / "last.pt", weights_only=True) for root in (full, resumed)]
        for key in a["weights"]:
            torch.testing.assert_close(a["weights"][key], b["weights"][key], rtol=0, atol=0)
        for key in ("input_mean", "input_scale", "target_mean", "target_scale"):
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        assert a["generated_episodes"] == b["generated_episodes"]
        assert (resumed / stage / "loss.png").exists()
    assert json.loads((resumed / "status.json").read_text())["stage"] == "complete"
    assert json.loads((resumed / "training_summary.json").read_text())["unique_transitions"] == 400
    saved = torch.load(resumed / "recursive/last.pt", weights_only=True)
    assert "Unweighted" in saved["provenance"]["loss"]
    prediction = load_predictor(resumed / "recursive/model.pt", 5)
    assert np.isfinite(prediction(np.array([0, 0, 0, 0, .4, .2, .035, 0, 1, 0, 0, 0, 0]), np.zeros(2))).all()
    output = capsys.readouterr()
    assert "epoch=" not in output.out and "\r" not in output.err


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware required")
def test_online_cuda_to_cpu_continuation(tmp_path):
    boundary = tmp_path / "boundary.npz"
    boundary_fixture(boundary)
    config = replace(tiny_config(), one_step_epochs=1)
    run(tmp_path / "gpu", config, "cuda", 1, boundary, evaluate=False)
    run(tmp_path / "cpu", replace(config, recursive_epochs=2), "cpu", 1, boundary, tmp_path / "gpu", evaluate=False)
    saved = torch.load(tmp_path / "cpu/recursive/last.pt", weights_only=True)
    assert saved["epoch"] == 2 and all(value.device.type == "cpu" for value in saved["weights"].values())
