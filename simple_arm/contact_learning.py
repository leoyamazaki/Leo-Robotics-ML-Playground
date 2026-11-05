"""Collect and learn arm–cube contact transitions at 20 and 100 ms."""

import argparse
import hashlib
import json
from pathlib import Path
import time

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
import torch
from torch import nn

from .arm_learning import normalization, load_predictor as load_arm_predictor
from .contact_simulation import ContactSimulation, link_lengths, model_hash


KINDS = ["free", "approach", "push_release", "sweep", "reversal", "near_contact"]
LABELS = ["no_contact", "onset", "sustained", "release", "sliding"]
BASE_DT = 0.02


def features(states, actions, lengths):
    result = states.copy()
    angle = states[..., 0]
    total = angle + states[..., 1]
    result[..., 4] -= lengths[0] * np.cos(angle) + lengths[1] * np.cos(total)
    result[..., 5] -= lengths[0] * np.sin(angle) + lengths[1] * np.sin(total)
    return np.concatenate([result, actions], axis=-1)


def delta_target(current, following):
    yaw = np.arctan2(following[..., 7] * current[..., 8] - following[..., 8] * current[..., 7], following[..., 8] * current[..., 8] + following[..., 7] * current[..., 7])
    return np.concatenate([following[..., :7] - current[..., :7], yaw[..., None], following[..., 9:] - current[..., 9:]], axis=-1)


def apply_delta(current, delta):
    result = current.copy()
    result[..., :7] += delta[..., :7]
    sine, cosine = np.sin(delta[..., 7]), np.cos(delta[..., 7])
    result[..., 7] = current[..., 7] * cosine + current[..., 8] * sine
    result[..., 8] = current[..., 8] * cosine - current[..., 7] * sine
    result[..., 9:] += delta[..., 8:]
    return result


def collect(path, episodes=360, seed=91):
    if episodes < 30:
        raise ValueError("At least 30 episodes required for stratified splitting")
    generator = np.random.default_rng(seed)
    simulation = ContactSimulation()
    lengths = link_lengths()
    states = np.empty((episodes, 201, 13))
    actions = np.empty((episodes, 200, 2))
    contacts = np.empty((episodes, 200, 10), dtype=bool)
    starting_contact = np.zeros((episodes, 200), dtype=bool)
    kinds = np.arange(episodes) % len(KINDS)
    for episode, kind in enumerate(kinds):
        arm = generator.uniform([-0.65, -0.9], [0.65, 0.9])
        direction = generator.choice([-1, 1])
        yaw = generator.uniform(-np.pi, np.pi)
        tip = np.array([lengths[0] * np.cos(arm[0]) + lengths[1] * np.cos(arm.sum()), lengths[0] * np.sin(arm[0]) + lengths[1] * np.sin(arm.sum())])
        tangent = direction * np.array([-tip[1], tip[0]]) / np.linalg.norm(tip)
        local = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]]) @ tangent
        distance = 0.035 * np.abs(local).sum() + 0.022 + (0.004 if kind == 5 else generator.uniform(0.025, 0.06))
        cube = tip + tangent * distance if kind else np.array([0.8, 0.8])
        velocity = np.array([direction * generator.uniform(0.1, 0.5), generator.uniform(-0.15, 0.15)]) if episode % 3 else None
        states[episode, 0] = simulation.initialize(arm, cube, yaw, velocity)
        speed = generator.uniform(0.25, 0.85)
        free_target = arm.copy()
        for step_index in range(200):
            if step_index % 5 == 0:
                elapsed = step_index * BASE_DT
                if kind == 0:
                    if step_index % 20 == 0:
                        free_target = arm + generator.uniform(-0.5, 0.5, 2)
                    target = free_target
                elif kind == 1:
                    target = arm + [direction * 0.02, 0]
                elif kind == 4 and 0.8 < elapsed < 2.4:
                    target = arm + [direction * (0.55 if int(elapsed / 0.3) % 2 else -0.15), 0]
                elif elapsed < (2.5 if kind == 3 else 1.6):
                    target = arm + [direction * min(0.85, speed * (elapsed + 0.2)), 0]
                else:
                    target = arm + [-direction * 0.25, 0.1 * direction]
                target = np.clip(target, simulation.action_limits[:, 0] + 0.1, simulation.action_limits[:, 1] - 0.1)
            actions[episode, step_index] = target
            starting_contact[episode, step_index] = simulation.contact()
            for substep in range(10):
                simulation.step(target)
                contacts[episode, step_index, substep] = simulation.contact()
            states[episode, step_index + 1] = simulation.extract()
        if (episode + 1) % 60 == 0:
            print(f"Collected {episode + 1}/{episodes} episodes", flush=True)
    split = np.empty(episodes, dtype=int)
    for kind in range(len(KINDS)):
        indices = generator.permutation(np.flatnonzero(kinds == kind))
        first, second = int(len(indices) * 0.7), int(len(indices) * 0.85)
        split[indices[:first]], split[indices[first:second]], split[indices[second:]] = 0, 1, 2
    metadata = {"base_dt": BASE_DT, "seed": seed, "episodes": episodes, "model_hash": model_hash(), "lengths": lengths, "kinds": KINDS,
                "data_sha256": hashlib.sha256(states.tobytes() + actions.tobytes() + split.tobytes()).hexdigest(),
                "state": "arm q2, arm v2, cube XYZ3, sin_yaw, cos_yaw, cube world v3, yaw_rate",
                "action": "absolute shoulder/elbow position targets in radians; held 100 ms",
                "physics_dt": 0.002, "mass_kg": 0.1, "friction": 0.7}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, states=states, actions=actions, contacts=contacts, starting_contact=starting_contact, split=split, kinds=kinds, metadata=json.dumps(metadata))
    dataset = load_dataset(path)
    coverage = {str(stride): {name: int(mask.sum()) for name, mask in labels(dataset, stride).items()} for stride in (1, 5)}
    summary = {"episodes": episodes, "transitions_20ms": episodes * 200, "episodes_per_split": np.bincount(split).tolist(), "coverage": coverage,
               "moving_episodes": int(np.count_nonzero(np.linalg.norm(states[:, -1, 4:6] - states[:, 0, 4:6], axis=-1) > 0.02)), "metadata": metadata}
    path.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return dataset


def load_dataset(path):
    with np.load(path, allow_pickle=False) as archive:
        dataset = {key: archive[key].copy() for key in ("states", "actions", "contacts", "starting_contact", "split", "kinds")}
        dataset["metadata"] = json.loads(str(archive["metadata"]))
    if dataset["metadata"]["model_hash"] != model_hash():
        raise ValueError("Contact model changed")
    if not np.isfinite(dataset["states"]).all() or not np.isfinite(dataset["actions"]).all():
        raise ValueError("Nonfinite data")
    states, actions, split = dataset["states"], dataset["actions"], dataset["split"]
    if states.shape != (len(states), 201, 13) or actions.shape != (len(states), 200, 2):
        raise ValueError("Invalid trajectory shape")
    if split.shape != (len(states),) or set(split.tolist()) != {0, 1, 2}:
        raise ValueError("Invalid episode splits")
    fingerprint = hashlib.sha256(states.tobytes() + actions.tobytes() + split.tobytes()).hexdigest()
    if dataset["metadata"].get("data_sha256", fingerprint) != fingerprint:
        raise ValueError("Dataset fingerprint mismatch")
    return dataset


def labels(dataset, stride):
    episodes, steps, substeps = dataset["contacts"].shape
    contact = dataset["contacts"].reshape(episodes, steps // stride, substeps * stride)
    start = dataset["starting_contact"][:, ::stride]
    before = np.concatenate([start[..., None], contact[..., :-1]], axis=-1)
    onset = np.any(contact & ~before, axis=-1)
    release = np.any(~contact & before, axis=-1)
    speed = np.linalg.norm(dataset["states"][:, ::stride, 9:11], axis=-1)
    sliding = ~contact.any(axis=-1) & (np.maximum(speed[:, :-1], speed[:, 1:]) > 0.005)
    active = contact.any(axis=-1)
    previous_active = np.concatenate([start[:, :1], active[:, :-1]], axis=1)
    return {"no_contact": ~active & ~sliding, "onset": onset,
            "sustained": previous_active & active, "release": release, "sliding": sliding}


def make_model(hidden_width=128):
    if not isinstance(hidden_width, int) or isinstance(hidden_width, bool) or hidden_width < 1:
        raise ValueError("hidden_width must be a positive integer")
    return nn.Sequential(nn.Linear(15, hidden_width), nn.SiLU(), nn.Linear(hidden_width, hidden_width), nn.SiLU(), nn.Linear(hidden_width, hidden_width), nn.SiLU(), nn.Linear(hidden_width, 12)).cpu()


def training_arrays(dataset, stride, split):
    selected = dataset["split"] == split
    states = dataset["states"][selected, ::stride]
    actions = dataset["actions"][selected, ::stride]
    inputs = features(states[:, :-1], actions, dataset["metadata"]["lengths"]).reshape(-1, 15)
    targets = delta_target(states[:, :-1], states[:, 1:]).reshape(-1, 12)
    contact = dataset["contacts"].reshape(len(dataset["states"]), -1, stride * 10)[selected].any(axis=-1).reshape(-1)
    weights = np.where(contact, 5.0, 1.0)
    return torch.tensor(inputs, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32), torch.tensor(weights, dtype=torch.float32)


def train(dataset, output, stride, epochs=100):
    torch.manual_seed(91)
    inputs, targets, weights = training_arrays(dataset, stride, 0)
    validation_inputs, validation_targets, _ = training_arrays(dataset, stride, 1)
    input_mean, input_scale = normalization(inputs)
    target_mean, target_scale = normalization(targets)
    input_scale = torch.maximum(input_scale, torch.tensor([0.1, 0.1, 0.5, 0.5, 0.03, 0.03, 0.01, 0.2, 0.2, 0.05, 0.05, 0.05, 0.2, 0.1, 0.1]))
    target_scale = torch.maximum(target_scale, torch.tensor([0.01, 0.01, 0.1, 0.1, 0.001, 0.001, 0.001, 0.01, 0.01, 0.01, 0.01, 0.05]))
    inputs, targets = (inputs - input_mean) / input_scale, (targets - target_mean) / target_scale
    validation_inputs = (validation_inputs - input_mean) / input_scale
    validation_targets = (validation_targets - target_mean) / target_scale
    model = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    best, history = float("inf"), []
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for indices in torch.randperm(len(inputs)).split(256):
            optimizer.zero_grad()
            loss = (((model(inputs[indices]) - targets[indices]) ** 2).mean(dim=1) * weights[indices]).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            total += loss.item() * len(indices)
        model.eval()
        with torch.no_grad():
            validation_loss = ((model(validation_inputs) - validation_targets) ** 2).mean().item()
        if not np.isfinite(validation_loss):
            raise RuntimeError("Nonfinite validation loss")
        history.append([total / len(inputs), validation_loss])
        if validation_loss < best:
            best = validation_loss
            torch.save({"weights": model.state_dict(), "input_mean": input_mean, "input_scale": input_scale,
                        "target_mean": target_mean, "target_scale": target_scale, "stride": stride,
                        "metadata": dataset["metadata"], "epoch": epoch + 1}, output / "model.pt")
        if (epoch + 1) % 20 == 0:
            print(f"dt={BASE_DT * stride:.2f} epoch={epoch + 1} weighted_train={history[-1][0]:.5f} validation={validation_loss:.5f}", flush=True)
    (output / "history.json").write_text(json.dumps(history), encoding="utf-8")
    figure, axes = new_plot()
    axes.semilogy(history)
    axes.legend(["Contact-weighted train", "Unweighted validation"])
    axes.set(xlabel="Epoch", ylabel="Normalized MSE (different weighting)")
    figure.savefig(output / "loss.png")
    return load_predictor(output / "model.pt")


def load_predictor(path, expected_stride=None):
    checkpoint = torch.load(path, weights_only=True, map_location="cpu")
    if checkpoint["metadata"]["model_hash"] != model_hash():
        raise ValueError("Contact checkpoint/model mismatch")
    if expected_stride is not None and checkpoint["stride"] != expected_stride:
        raise ValueError("Checkpoint prediction interval mismatch")
    model = make_model(checkpoint.get("hidden_width", 128)).eval()
    model.load_state_dict(checkpoint["weights"])

    def predict(current, action):
        inputs = torch.tensor(features(current, action, checkpoint["metadata"]["lengths"]), dtype=torch.float32)
        with torch.no_grad():
            delta = model((inputs - checkpoint["input_mean"]) / checkpoint["input_scale"]) * checkpoint["target_scale"] + checkpoint["target_mean"]
        result = apply_delta(current, delta.numpy())
        if not np.isfinite(result).all():
            raise RuntimeError("Nonfinite contact prediction")
        return result

    return predict


def differentiable_next(model, current, action, checkpoint):
    first, second = checkpoint["metadata"]["lengths"]
    tip = torch.stack([first * torch.cos(current[..., 0]) + second * torch.cos(current[..., 0] + current[..., 1]), first * torch.sin(current[..., 0]) + second * torch.sin(current[..., 0] + current[..., 1])], dim=-1)
    inputs = torch.cat([current[..., :4], current[..., 4:6] - tip, current[..., 6:], action], dim=-1)
    delta = model((inputs - checkpoint["input_mean"]) / checkpoint["input_scale"]) * checkpoint["target_scale"] + checkpoint["target_mean"]
    sine, cosine = torch.sin(delta[..., 7]), torch.cos(delta[..., 7])
    orientation = torch.stack([current[..., 7] * cosine + current[..., 8] * sine, current[..., 8] * cosine - current[..., 7] * sine], dim=-1)
    return torch.cat([current[..., :7] + delta[..., :7], orientation, current[..., 9:] + delta[..., 8:]], dim=-1)


def train_multistep(dataset, source, output, epochs=80, *, shuffle_seed=None, validation_every=5):
    """Refine the 100 ms model using predicted-state feedback and validation rollouts."""
    if epochs < 1 or validation_every < 1:
        raise ValueError("epochs and validation_every must be positive")
    torch.manual_seed(93)
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    if checkpoint["stride"] != 5 or checkpoint["metadata"] != dataset["metadata"]:
        raise ValueError("Refinement requires matching 100 ms dataset/checkpoint")
    model = make_model(checkpoint.get("hidden_width", 128))
    model.load_state_dict(checkpoint["weights"])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    states = torch.tensor(dataset["states"][dataset["split"] == 0, ::5], dtype=torch.float32)
    actions = torch.tensor(dataset["actions"][dataset["split"] == 0, ::5], dtype=torch.float32)
    validation_states = torch.tensor(dataset["states"][dataset["split"] == 1, ::5], dtype=torch.float32)
    validation_actions = torch.tensor(dataset["actions"][dataset["split"] == 1, ::5], dtype=torch.float32)
    scales = torch.tensor([1, 1, 3, 3, 0.1, 0.1, 0.01, 0.5, 0.5, 0.5, 0.5, 0.1, 2])
    contacts = dataset["contacts"].reshape(len(dataset["states"]), 40, 50).any(axis=-1)[dataset["split"] == 0]
    windows = [(episode, start) for episode in range(len(states)) for start in range(31)]
    windows = torch.tensor(windows)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    best, history = float("inf"), []
    shuffle = torch.Generator().manual_seed(shuffle_seed) if shuffle_seed is not None else None
    epoch_timings, updates = [], 0
    started = time.perf_counter()

    def validate():
        model.eval()
        with torch.no_grad():
            current = validation_states[:, 0]
            losses = []
            for offset in range(40):
                current = differentiable_next(model, current, validation_actions[:, offset], checkpoint)
                losses.append(((current - validation_states[:, offset + 1]) / scales).square().mean())
            return torch.stack(losses).mean().item()

    for epoch in range(epochs + 1):
        epoch_started = time.perf_counter()
        if epoch:
            model.train()
            for indices in torch.randperm(len(windows), generator=shuffle).split(256):
                episode, start = windows[indices].T
                current = states[episode, start]
                loss = 0
                for offset in range(10):
                    current = differentiable_next(model, current, actions[episode, start + offset], checkpoint)
                    weight = torch.tensor(np.where(contacts[episode.numpy(), (start + offset).numpy()], 3.0, 1.0), dtype=torch.float32)
                    loss = loss + ((((current - states[episode, start + offset + 1]) / scales).square().mean(dim=-1)) * weight).mean() / 10
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite recursive training loss")
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                updates += 1
        if epoch % validation_every == 0 or epoch == epochs:
            validation_loss = validate()
            history.append({"epoch": epoch, "validation_recursive_scaled_mse": validation_loss})
            if np.isfinite(validation_loss) and validation_loss < best:
                best = validation_loss
                saved = dict(checkpoint, weights=model.state_dict(), refinement_epoch=epoch,
                             refinement="10-step recursive training; 40-step validation",
                             training_method="one_step_plus_recursive", recursive_epoch_cap=epochs,
                             recursive_shuffle_seed=shuffle_seed, recursive_validation_every=validation_every)
                torch.save(saved, output / "model.pt")
            print(f"recursive epoch={epoch} validation_rollout_loss={validation_loss:.5f}", flush=True)
        if epoch:
            epoch_timings.append(time.perf_counter() - epoch_started)
    timing = {"epochs_run": epochs, "optimizer_updates": updates,
              "updates_per_epoch": (len(windows) + 255) // 256,
              "wall_seconds": time.perf_counter() - started, "epoch_seconds": epoch_timings,
              "mean_epoch_seconds": float(np.mean(epoch_timings)),
              "shuffle_seed": shuffle_seed, "validation_every": validation_every}
    (output / "refinement_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output / "refinement_training.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
    return timing


def physical_errors(predicted, actual, initial):
    difference = predicted - actual
    axes = tuple(range(difference.ndim - 1))
    yaw = np.arctan2(predicted[..., 7] * actual[..., 8] - predicted[..., 8] * actual[..., 7], predicted[..., 8] * actual[..., 8] + predicted[..., 7] * actual[..., 7])
    actual_motion = actual[..., 4:6] - initial[..., 4:6]
    predicted_motion = predicted[..., 4:6] - initial[..., 4:6]
    moved = np.linalg.norm(actual_motion, axis=-1) > 0.002
    false_motion = np.linalg.norm(predicted_motion, axis=-1) > 0.002
    cosine = np.sum(actual_motion * predicted_motion, axis=-1) / np.maximum(np.linalg.norm(actual_motion, axis=-1) * np.linalg.norm(predicted_motion, axis=-1), 1e-12)
    return {
        "samples": int(np.prod(actual.shape[:-1])),
        "arm_position_rmse_deg": np.rad2deg(np.sqrt(np.mean(difference[..., :2] ** 2, axis=axes))).tolist(),
        "arm_velocity_rmse_rad_s": np.sqrt(np.mean(difference[..., 2:4] ** 2, axis=axes)).tolist(),
        "cube_position_rmse_m": np.sqrt(np.mean(difference[..., 4:7] ** 2, axis=axes)).tolist(),
        "cube_yaw_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(yaw ** 2)))),
        "cube_velocity_rmse_m_s": np.sqrt(np.mean(difference[..., 9:12] ** 2, axis=axes)).tolist(),
        "cube_yaw_rate_rmse_rad_s": float(np.sqrt(np.mean(difference[..., 12] ** 2))),
        "displacement_vector_rmse_m": float(np.sqrt(np.mean(np.sum((predicted_motion - actual_motion) ** 2, axis=-1)))),
        "displacement_magnitude_mae_m": float(np.mean(np.abs(np.linalg.norm(predicted_motion, axis=-1) - np.linalg.norm(actual_motion, axis=-1)))),
        "motion_samples": int(moved.sum()),
        "mean_direction_cosine_when_moving": float(cosine[moved].mean()) if moved.any() else None,
        "false_motion_fraction_when_stationary": float(false_motion[~moved].mean()) if (~moved).any() else None,
        "missed_motion_fraction": float((~false_motion[moved]).mean()) if moved.any() else None,
    }


def new_plot():
    figure = Figure(figsize=(9, 4), layout="constrained")
    FigureCanvasAgg(figure)
    return figure, figure.subplots()


def rollout(predict, initial, actions):
    trajectory = [initial.copy()]
    for step_index in range(actions.shape[1]):
        trajectory.append(predict(trajectory[-1], actions[:, step_index]))
    return np.stack(trajectory, axis=1)


def evaluate(dataset, predict, output, stride):
    selected = dataset["split"] == 2
    states, actions = dataset["states"][selected, ::stride], dataset["actions"][selected, ::stride]
    dt = stride * BASE_DT
    masks = {name: mask[selected] for name, mask in labels(dataset, stride).items()}
    arm_predict = load_arm_predictor("runs/arm_dynamics_broad/model.pt") if stride == 5 else None

    def stationary(current, action):
        result = current.copy()
        if arm_predict is not None:
            result[..., :4] = arm_predict(current[..., :4], action)
        return result

    def constant_velocity(current, action):
        result = stationary(current, action)
        result[..., 4:7] += current[..., 9:12] * dt
        delta = np.zeros((*current.shape[:-1], 12))
        delta[..., 7] = current[..., 12] * dt
        return apply_delta(result, delta)

    report = {"dt": dt, "evaluation_metadata": dataset["metadata"], "baseline_arm_mode": "original 100 ms arm model" if stride == 5 else "frozen (cube-only baseline comparison)",
              "label_definition": "onset/release: any substep transition; sustained: contact activity in consecutive windows; masks may overlap",
              "one_step": {}, "conditioned": {}, "rollout_conditioned": {}, "rollout": {}, "endpoint": {}, "moving_episode_endpoint": {}}
    trajectories = {}
    for name, predictor in (("learned", predict), ("stationary", stationary), ("constant_velocity", constant_velocity)):
        one_step = predictor(states[:, :-1], actions)
        report["one_step"][name] = physical_errors(one_step, states[:, 1:], states[:, :-1])
        report["conditioned"][name] = {label: physical_errors(one_step[mask], states[:, 1:][mask], states[:, :-1][mask]) if mask.any() else {"samples": 0} for label, mask in masks.items()}
        predicted = rollout(predictor, states[:, 0], actions)
        trajectories[name] = predicted
        report["rollout"][name] = {str(seconds): physical_errors(predicted[:, :round(seconds / dt) + 1][:, 1:], states[:, :round(seconds / dt) + 1][:, 1:], np.broadcast_to(states[:, :1], predicted[:, 1:round(seconds / dt) + 1].shape)) for seconds in (1, 2, 4)}
        report["endpoint"][name] = physical_errors(predicted[:, -1], states[:, -1], states[:, 0])
        accumulated_initial = np.broadcast_to(states[:, :1], states[:, 1:].shape)
        report["rollout_conditioned"][name] = {label: physical_errors(predicted[:, 1:][mask], states[:, 1:][mask], accumulated_initial[mask]) if mask.any() else {"samples": 0} for label, mask in masks.items()}
        moved = np.linalg.norm(states[:, -1, 4:6] - states[:, 0, 4:6], axis=-1) > 0.02
        report["moving_episode_endpoint"][name] = physical_errors(predicted[moved, -1], states[moved, -1], states[moved, 0]) if moved.any() else {"samples": 0}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    np.savez_compressed(output / "test_rollouts.npz", actual=states, actions=actions, **trajectories)
    figure, axes = new_plot()
    for name, trajectory in trajectories.items():
        error = np.sqrt(np.mean(np.sum((trajectory[:, 1:, 4:6] - states[:, 1:, 4:6]) ** 2, axis=-1), axis=0))
        axes.plot(np.arange(1, len(error) + 1) * dt, error, label=name)
    axes.set(xlabel="Autoregressive horizon (s)", ylabel="Cube XY vector RMSE (m)")
    axes.legend()
    figure.savefig(output / "horizon_errors.png")
    moving = np.flatnonzero(np.linalg.norm(states[:, -1, 4:6] - states[:, 0, 4:6], axis=-1) > 0.02)
    scores = np.mean(np.sum((trajectories["learned"][..., 4:6] - states[..., 4:6]) ** 2, axis=-1), axis=1)
    if len(moving):
        for tag, index in (("best_push", moving[np.argmin(scores[moving])]), ("worst_push", moving[np.argmax(scores[moving])])):
            figure = Figure(figsize=(12, 12), layout="constrained")
            FigureCanvasAgg(figure)
            panels = figure.subplots(5, 2)
            times = np.arange(states.shape[1]) * dt
            for joint in range(2):
                for name, trajectory in (("actual", states), ("learned", trajectories["learned"])):
                    panels[0, joint].plot(times, trajectory[index, :, joint], label=name)
                    panels[1, joint].plot(times, trajectory[index, :, 4 + joint], label=name)
                    panels[2, joint].plot(times, trajectory[index, :, 9 + joint], label=name)
                panels[0, joint].set(ylabel="Arm angle (rad)")
                panels[1, joint].set(ylabel="Cube position (m)")
                panels[2, joint].set(ylabel="Cube velocity (m/s)")
                panels[3, joint].plot(times[1:], trajectories["learned"][index, 1:, 4 + joint] - states[index, 1:, 4 + joint], label="position error")
                contact = dataset["contacts"][selected][index].reshape(-1, stride * 10).any(axis=-1)
                contact_axes = panels[3, joint].twinx()
                contact_axes.fill_between(times[1:], 0, contact.astype(int), alpha=0.15, color="red", label="actual contact")
                contact_axes.set_ylim(0, 3)
            panels[4, 0].plot(states[index, :, 4], states[index, :, 5], label="actual XY")
            for name, trajectory in trajectories.items():
                panels[4, 0].plot(trajectory[index, :, 4], trajectory[index, :, 5], label=name)
            panels[4, 0].set(xlabel="X (m)", ylabel="Y (m)", aspect="equal")
            panels[4, 1].plot(times, np.unwrap(np.arctan2(states[index, :, 7], states[index, :, 8])), label="actual yaw")
            panels[4, 1].plot(times, np.unwrap(np.arctan2(trajectories["learned"][index, :, 7], trajectories["learned"][index, :, 8])), label="predicted yaw")
            for panel in panels.flat:
                panel.legend()
            figure.suptitle(f"{tag}: held-out episode {np.flatnonzero(selected)[index]}, dt={dt}")
            figure.savefig(output / f"{tag}.png")
    print(f"Evaluation dt={dt}: " + json.dumps(report["endpoint"]), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["collect", "train", "refine", "evaluate", "run"])
    parser.add_argument("--dataset", type=Path, default=Path("data/arm_cube_dynamics.npz"))
    parser.add_argument("--output", type=Path, default=Path("runs/arm_cube_dynamics"))
    parser.add_argument("--episodes", type=int, default=360)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=91)
    parser.add_argument("--stride", type=int, choices=[1, 5], default=5)
    parser.add_argument("--source", type=Path, default=Path("runs/arm_cube_dynamics/model.pt"))
    parser.add_argument("--checkpoint", type=Path, help="Evaluation checkpoint, independent of the output directory")
    parser.add_argument("--all-episodes", action="store_true", help="Evaluate every episode of an independent dataset")
    args = parser.parse_args()
    torch.set_num_threads(2)
    dataset = collect(args.dataset, args.episodes, args.seed) if args.command in ("collect", "run") else load_dataset(args.dataset)
    if args.command in ("train", "run"):
        train(dataset, args.output, args.stride, args.epochs)
    if args.command == "refine":
        train_multistep(dataset, args.source, args.output, args.epochs)
    if args.command in ("evaluate", "run"):
        checkpoint_path = args.checkpoint or args.output / "model.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint["metadata"] != dataset["metadata"] and not args.all_episodes:
            raise ValueError("Checkpoint/dataset provenance mismatch")
        if args.all_episodes:
            if checkpoint["metadata"]["seed"] == dataset["metadata"]["seed"]:
                raise ValueError("Independent evaluation requires a different collection seed")
            dataset = dict(dataset, split=np.full_like(dataset["split"], 2))
        evaluate(dataset, load_predictor(checkpoint_path, args.stride), args.output, args.stride)


if __name__ == "__main__":
    main()
