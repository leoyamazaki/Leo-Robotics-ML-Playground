import numpy as np
import pytest
import torch

from simple_arm.contact_learning import (
    apply_delta, collect, delta_target, features, labels, load_predictor,
    differentiable_next, make_model, rollout, train, training_arrays,
)
from simple_arm.contact_simulation import ContactSimulation


@pytest.fixture(scope="module")
def contact_dataset(tmp_path_factory):
    path = tmp_path_factory.mktemp("contact")
    return collect(path / "dataset.npz", episodes=30, seed=12), path


def test_cube_is_supported_mobile_and_has_friction():
    simulation = ContactSimulation()
    initial = simulation.initialize([0, 0], [0.52, 0.13], 0)
    assert initial.shape == (13,)
    assert initial[6] == pytest.approx(0.035, abs=0.002)
    assert simulation.model.body_mass[simulation.cube_id] == pytest.approx(0.1)
    assert np.all(simulation.model.body_inertia[simulation.cube_id] > 0)
    assert not simulation.contact()
    touched = False
    for _ in range(1000):
        simulation.step([0.65, 0])
        touched |= simulation.contact()
    pushed = simulation.extract()
    assert touched
    assert np.linalg.norm(pushed[4:6] - initial[4:6]) > 0.02
    observation = simulation.get_observation()
    np.testing.assert_allclose(pushed[4:7], observation["cube_position"])
    np.testing.assert_allclose(pushed[9:12], observation["cube_linear_velocity"])
    simulation.initialize([0, 0], [0.8, 0.8], 0)
    simulation.data.qvel[simulation.cube_v[0]] = 0.5
    import mujoco
    mujoco.mj_forward(simulation.model, simulation.data)
    for _ in range(250):
        simulation.step([0, 0])
    assert abs(simulation.extract()[9]) < 0.01


def test_yaw_targets_wrap_and_residual_reconstruction():
    current = np.zeros((2, 13))
    following = np.ones((2, 13)) * 0.1
    yaw = np.array([np.pi - 0.01, -np.pi + 0.01])
    next_yaw = np.array([-np.pi + 0.02, np.pi - 0.02])
    current[:, 7:9] = np.stack([np.sin(yaw), np.cos(yaw)], axis=-1)
    following[:, 7:9] = np.stack([np.sin(next_yaw), np.cos(next_yaw)], axis=-1)
    delta = delta_target(current, following)
    np.testing.assert_allclose(delta[:, 7], [0.03, -0.03])
    np.testing.assert_allclose(apply_delta(current, delta), following, atol=1e-12)


def test_contact_collection_coverage_split_and_reproducibility(contact_dataset):
    dataset, path = contact_dataset
    repeated = collect(path / "repeated.npz", episodes=30, seed=12)
    for key in ("states", "actions", "contacts", "starting_contact", "split"):
        np.testing.assert_array_equal(dataset[key], repeated[key])
    assert dataset["states"].shape == (30, 201, 13)
    for split in range(3):
        assert set(dataset["kinds"][dataset["split"] == split]) == set(range(6))
    for mask in labels(dataset, 1).values():
        assert mask.sum() > 0
    np.testing.assert_allclose(np.linalg.norm(dataset["states"][..., 7:9], axis=-1), 1)
    assert np.max(np.abs(dataset["states"][:, 0, 2:4])) > 0.1
    actions = dataset["actions"].reshape(30, 40, 5, 2)
    np.testing.assert_array_equal(actions, np.broadcast_to(actions[:, :, :1], actions.shape))


def test_inputs_do_not_include_contact_labels(contact_dataset):
    dataset, _ = contact_dataset
    inputs, targets, weights = training_arrays(dataset, 5, 0)
    changed = dict(dataset, contacts=~dataset["contacts"])
    other_inputs, other_targets, other_weights = training_arrays(changed, 5, 0)
    torch.testing.assert_close(inputs, other_inputs)
    torch.testing.assert_close(targets, other_targets)
    assert inputs.shape[1] == 15 and targets.shape[1] == 12
    states = dataset["states"][dataset["split"] == 0, ::5]
    torch.testing.assert_close(targets, torch.tensor(delta_target(states[:, :-1], states[:, 1:]).reshape(-1, 12), dtype=torch.float32))


def test_contact_rollout_is_autoregressive():
    initial = np.zeros((2, 13))
    initial[:, 8] = 1
    actions = np.ones((2, 200, 2)) / 8
    calls = []

    def predictor(current, action):
        calls.append(current.copy())
        delta = np.zeros((2, 12))
        delta[:, 4:6] = action
        return apply_delta(current, delta)

    predicted = rollout(predictor, initial, actions)
    np.testing.assert_array_equal(np.stack(calls, axis=1), predicted[:, :-1])
    np.testing.assert_array_equal(predicted[:, -1, 4:6], np.full((2, 2), 25))


def test_contact_checkpoint_normalization_and_interval(contact_dataset):
    dataset, path = contact_dataset
    torch.set_num_threads(2)
    predictor = train(dataset, path / "run", stride=5, epochs=2)
    checkpoint = torch.load(path / "run/model.pt", weights_only=True)
    inputs, _, _ = training_arrays(dataset, 5, 0)
    torch.testing.assert_close(checkpoint["input_mean"], inputs.mean(0))
    current, action = dataset["states"][0, 0], dataset["actions"][0, 0]
    predicted = predictor(current, action)
    assert np.isfinite(predicted).all()
    assert np.linalg.norm(predicted[7:9]) == pytest.approx(1)
    model = make_model().eval()
    model.load_state_dict(checkpoint["weights"])
    tensor_prediction = differentiable_next(model, torch.tensor(current, dtype=torch.float32), torch.tensor(action, dtype=torch.float32), checkpoint)
    np.testing.assert_allclose(tensor_prediction.detach().numpy(), predicted, atol=1e-6)
    tensor_prediction.square().mean().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    with pytest.raises(ValueError, match="interval"):
        load_predictor(path / "run/model.pt", expected_stride=1)
