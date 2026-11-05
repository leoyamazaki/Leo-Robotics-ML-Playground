import numpy as np
import pytest
import torch

from simple_arm.capacity_screen import file_hash, normalized_losses, prepare, train_width
from simple_arm.contact_learning import differentiable_next, load_predictor, make_model, training_arrays, train_multistep
from simple_arm.contact_simulation import model_hash


@pytest.mark.parametrize("width,parameters", [(128, 36620), (256, 138764), (512, 539660)])
def test_capacity_changes_only_hidden_width(width, parameters):
    model = make_model(width)
    assert sum(p.numel() for p in model.parameters()) == parameters
    assert model(torch.zeros(2, 15)).shape == (2, 12)
    assert sum(isinstance(layer, torch.nn.SiLU) for layer in model) == 3


def fixture_arrays():
    generator = np.random.default_rng(45)
    states = generator.normal(0, .1, (3, 201, 13))
    yaw = generator.uniform(-np.pi, np.pi, (3, 201))
    states[..., 7], states[..., 8] = np.sin(yaw), np.cos(yaw)
    data = {"states": states, "actions": generator.normal(0, .1, (3, 200, 2)),
            "contacts": np.zeros((3, 200, 10), dtype=bool), "split": np.arange(3),
            "metadata": {"seed": 45, "lengths": [.3, .25], "model_hash": model_hash()}}
    x, y, _ = training_arrays(data, 5, 0)
    reference = {"metadata": data["metadata"], "stride": 5, "input_mean": x.mean(0),
                 "target_mean": y.mean(0), "input_scale": x.std(0).clamp_min(.01),
                 "target_scale": y.std(0).clamp_min(.01)}
    return data, reference


def test_identical_update_caps_and_width_aware_checkpoint_roundtrip(tmp_path):
    torch.set_num_threads(2)
    data, reference = fixture_arrays()
    train, validation = prepare(data, reference)
    for width in (16, 32):
        path = tmp_path / str(width)
        stats = train_width(width, reference, train, validation, path, epochs=2)
        assert stats["optimizer_updates"] == 2
        assert stats["epochs_run"] == 2
        checkpoint = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
        assert checkpoint["hidden_width"] == width
        for name in ("input_mean", "input_scale", "target_mean", "target_scale"):
            torch.testing.assert_close(checkpoint[name], reference[name])
        model = make_model(width)
        model.load_state_dict(checkpoint["weights"])
        assert np.isfinite(normalized_losses(model, validation)["unweighted"])
        state, action = data["states"][2, 0], data["actions"][2, 0]
        actual = load_predictor(path / "model.pt", 5)(state, action)
        expected = differentiable_next(model, torch.tensor(state, dtype=torch.float32), torch.tensor(action, dtype=torch.float32), checkpoint)
        np.testing.assert_allclose(actual, expected.detach().numpy(), atol=1e-6)
        with pytest.raises(FileExistsError):
            train_width(width, reference, train, validation, path, epochs=2)


def test_legacy_checkpoint_and_provenance_guard(tmp_path):
    data, reference = fixture_arrays()
    path = tmp_path / "legacy.pt"
    torch.save(dict(reference, weights=make_model().state_dict(), epoch=1), path)
    result = load_predictor(path, 5)(data["states"][0, 0], data["actions"][0, 0])
    assert result.shape == (13,) and np.isfinite(result).all()
    altered = dict(reference, metadata=dict(reference["metadata"], seed=46))
    with pytest.raises(ValueError, match="exact"):
        prepare(data, altered)
    for width in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            make_model(width)


def test_recursive_widths_equal_update_caps_and_preserve_sources(tmp_path):
    """Numerical fixtures only; no MuJoCo stepping or trajectory collection."""
    torch.set_num_threads(2)
    data, reference = fixture_arrays()
    for width in (16, 32):
        model = make_model(width)
        with torch.no_grad():
            model[-1].weight.zero_()
            model[-1].bias.zero_()
        source = tmp_path / f"source_{width}.pt"
        torch.save(dict(reference, weights=model.state_dict(), hidden_width=width, epoch=1), source)
        digest = file_hash(source)
        output = tmp_path / f"recursive_{width}"
        stats = train_multistep(data, source, output, epochs=1, shuffle_seed=93, validation_every=1)
        assert stats["optimizer_updates"] == stats["updates_per_epoch"] == 1
        assert len(stats["epoch_seconds"]) == 1
        assert stats["validation_every"] == 1
        checkpoint = torch.load(output / "model.pt", map_location="cpu", weights_only=True)
        assert checkpoint["hidden_width"] == width
        assert checkpoint["refinement_epoch"] in (0, 1)
        assert checkpoint["training_method"] == "one_step_plus_recursive"
        for name in ("input_mean", "input_scale", "target_mean", "target_scale"):
            torch.testing.assert_close(checkpoint[name], reference[name])
        assert file_hash(source) == digest
        result = load_predictor(output / "model.pt", 5)(data["states"][2, 0], data["actions"][2, 0])
        assert np.isfinite(result).all()
