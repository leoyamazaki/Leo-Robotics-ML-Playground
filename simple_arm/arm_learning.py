"""Collect, train, and evaluate a CPU model of contact-free arm dynamics."""

import argparse
import hashlib
import json
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import mujoco
import numpy as np
import torch
from torch import nn

from .simulation import ArmSimulation, MODEL_PATH


DT = 0.1
STATE_NAMES = ["shoulder_angle", "elbow_angle", "shoulder_velocity", "elbow_velocity"]
SCENARIOS = ["small_moves", "large_moves", "reversals", "both_joints"]
BROAD_SCENARIOS = SCENARIOS + ["irregular", "fast_reversals", "limit_holds", "limit_reversals"]


def state(simulation):
    observation = simulation.get_observation()
    return np.concatenate([observation["joint_positions"], observation["joint_velocities"]])


def collect(path, episodes=100, steps=50, seed=42, profile="original"):
    """Save whole episodes of 100 ms transitions and a deterministic split."""
    if profile not in ("original", "broad"):
        raise ValueError("Unknown collection profile")
    labels = BROAD_SCENARIOS if profile == "broad" else SCENARIOS
    if episodes < 5 * len(labels) or steps < 10:
        raise ValueError(f"Use at least {5 * len(labels)} episodes and 10 steps per episode")
    generator = np.random.default_rng(seed)
    simulation = ArmSimulation(frame_skip=50)
    states = np.empty((episodes, steps + 1, 4))
    actions = np.empty((episodes, steps, 2))
    scenarios = np.arange(episodes) % len(labels)
    cube_address = simulation.model.jnt_qposadr[simulation.model.joint("cube_free").id]
    for episode in range(episodes):
        simulation.reset()
        simulation.data.qpos[cube_address:cube_address + 3] = [1.2, 1.2, 0.12]
        initial = generator.uniform([-0.9, -1.2], [0.9, 1.2])
        scenario = scenarios[episode]
        signs = generator.choice([-1, 1], size=2) if profile == "broad" else None
        if profile == "broad" and scenario >= 4:
            initial = generator.uniform(simulation.action_limits[:, 0], simulation.action_limits[:, 1]) * 0.98
            if scenario >= 6:
                initial = signs * simulation.action_limits[:, 1] * generator.uniform(0.90, 0.995, 2)
        if profile == "broad" and episode % 3 != 0:
            simulation.data.qvel[simulation.qvel_indices] = generator.uniform([-5, -7], [5, 7])
        simulation.data.qpos[simulation.qpos_indices] = initial
        simulation.data.ctrl[:] = initial
        mujoco.mj_forward(simulation.model, simulation.data)
        target = initial.copy()
        next_change = 0
        states[episode, 0] = state(simulation)
        for step_index in range(steps):
            if step_index == next_change:
                hold = 5
                if scenario == 0:
                    target = initial + generator.uniform(-0.2, 0.2, 2)
                elif scenario == 1:
                    target = initial.copy()
                    moving_joint = (step_index // 5) % 2
                    target[moving_joint] = generator.uniform(-1.2, 1.2)
                elif scenario == 2:
                    target = np.array([0.9, -1.1]) * (-1 if (step_index // 5) % 2 else 1)
                elif scenario == 3:
                    target = generator.uniform([-1.2, -1.6], [1.2, 1.6])
                elif scenario == 4:
                    target = generator.uniform(simulation.action_limits[:, 0], simulation.action_limits[:, 1])
                    hold = int(generator.integers(1, 9))
                elif scenario == 5:
                    signs *= -1
                    target = signs * generator.uniform([0.6, 0.8], [1.5, 2.0])
                    hold = int(generator.integers(1, 5))
                else:
                    signs *= -1
                    target = signs * simulation.action_limits[:, 1] * generator.uniform(0.95, 1.0, 2)
                    if episode % 2 == 0:
                        target = signs * simulation.action_limits[:, 1]
                    hold = int(generator.integers(7, 14) if scenario == 6 else generator.integers(1, 5))
                next_change = step_index + hold
            target = np.clip(target, simulation.action_limits[:, 0], simulation.action_limits[:, 1])
            simulation.step(target)
            states[episode, step_index + 1] = state(simulation)
            actions[episode, step_index] = target
    split = np.empty(episodes, dtype=np.int64)
    for scenario in range(len(labels)):
        indices = generator.permutation(np.flatnonzero(scenarios == scenario))
        train_end = int(0.7 * len(indices))
        validation_end = max(train_end + 1, int(0.85 * len(indices)))
        split[indices[:train_end]] = 0
        split[indices[train_end:validation_end]] = 1
        split[indices[validation_end:]] = 2
    metadata = {
        "seed": seed, "episodes": episodes, "steps_per_episode": steps,
        "dt": DT, "state_order": STATE_NAMES,
        "action": "absolute shoulder, elbow position targets in radians",
        "state_units": ["rad", "rad", "rad/s", "rad/s"],
        "scenarios": labels, "split_names": ["train", "validation", "test"],
        "collection_profile": profile,
        "initial_velocity_range_rad_s": [[-5, 5], [-7, 7]] if profile == "broad" else [[0, 0], [0, 0]],
        "cube_initial_position": [1.2, 1.2, 0.12],
        "model_sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
        "mujoco_version": mujoco.__version__, "numpy_version": np.__version__,
        "data_sha256": hashlib.sha256(states.tobytes() + actions.tobytes()).hexdigest(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, states=states, actions=actions, split=split,
                        scenarios=scenarios, metadata=json.dumps(metadata))
    summary = {
        "metadata": metadata,
        "transitions": episodes * steps,
        "episodes_per_split": np.bincount(split, minlength=3).tolist(),
        "episodes_per_scenario": np.bincount(scenarios, minlength=len(labels)).tolist(),
        "state_min": states.min(axis=(0, 1)).tolist(),
        "state_max": states.max(axis=(0, 1)).tolist(),
        "action_min": actions.min(axis=(0, 1)).tolist(),
        "action_max": actions.max(axis=(0, 1)).tolist(),
    }
    path.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Collected {episodes * steps} transitions: {path}")
    return load_dataset(path)


def load_dataset(path):
    with np.load(path, allow_pickle=False) as archive:
        dataset = {name: archive[name].copy() for name in ("states", "actions", "split", "scenarios")}
        dataset["metadata"] = json.loads(str(archive["metadata"]))
    states, actions = dataset["states"], dataset["actions"]
    if states.ndim != 3 or states.shape[-1] != 4 or actions.shape != (states.shape[0], states.shape[1] - 1, 2):
        raise ValueError("Invalid state/action shapes")
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError("Dataset contains nonfinite values")
    if dataset["split"].shape != (len(states),) or set(dataset["split"].tolist()) != {0, 1, 2}:
        raise ValueError("Dataset must have nonempty, episode-level train/validation/test splits")
    if dataset["metadata"]["dt"] != DT or dataset["metadata"]["state_order"] != STATE_NAMES:
        raise ValueError("Incompatible state definition or prediction interval")
    if dataset["scenarios"].shape != (len(states),) or not np.isin(dataset["scenarios"], np.arange(len(dataset["metadata"]["scenarios"]))).all():
        raise ValueError("Invalid episode scenario labels")
    if hashlib.sha256(states.tobytes() + actions.tobytes()).hexdigest() != dataset["metadata"]["data_sha256"]:
        raise ValueError("Dataset content does not match its fingerprint")
    return dataset


def transitions(dataset, split):
    selected = dataset["split"] == split
    states = dataset["states"][selected]
    actions = dataset["actions"][selected]
    inputs = np.concatenate([states[:, :-1], actions], axis=-1).reshape(-1, 6)
    deltas = (states[:, 1:] - states[:, :-1]).reshape(-1, 4)
    return torch.tensor(inputs, dtype=torch.float32), torch.tensor(deltas, dtype=torch.float32)


def make_model():
    return nn.Sequential(nn.Linear(6, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 4)).cpu()


def normalization(values):
    return values.mean(0), values.std(0, unbiased=False).clamp_min(1e-6)


def train(dataset, output, epochs=150, seed=42):
    if epochs < 1:
        raise ValueError("epochs must be positive")
    torch.manual_seed(seed)
    inputs, deltas = transitions(dataset, 0)
    validation_inputs, validation_deltas = transitions(dataset, 1)
    input_mean, input_scale = normalization(inputs)
    delta_mean, delta_scale = normalization(deltas)
    inputs = (inputs - input_mean) / input_scale
    deltas = (deltas - delta_mean) / delta_scale
    validation_inputs = (validation_inputs - input_mean) / input_scale
    validation_deltas = (validation_deltas - delta_mean) / delta_scale
    model = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.MSELoss()
    history = []
    best_loss = float("inf")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for epoch in range(epochs):
        model.train()
        for indices in torch.randperm(len(inputs)).split(128):
            optimizer.zero_grad()
            loss = loss_function(model(inputs[indices]), deltas[indices])
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            training_loss = loss_function(model(inputs), deltas).item()
            validation_loss = loss_function(model(validation_inputs), validation_deltas).item()
        if not np.isfinite([training_loss, validation_loss]).all():
            raise RuntimeError("Nonfinite epoch loss")
        history.append([training_loss, validation_loss])
        if validation_loss < best_loss:
            best_loss = validation_loss
            torch.save({
                "weights": model.state_dict(), "input_mean": input_mean, "input_scale": input_scale,
                "delta_mean": delta_mean, "delta_scale": delta_scale,
                "metadata": dataset["metadata"], "epoch": epoch + 1, "seed": seed,
                "torch_version": str(torch.__version__),
            }, output / "model.pt")
        if (epoch + 1) % 25 == 0:
            print(f"epoch={epoch + 1} normalized MSE: train={training_loss:.5f}, validation={validation_loss:.5f}")
    (output / "history.json").write_text(json.dumps(history), encoding="utf-8")
    figure, axes = new_figure()
    axes.semilogy(np.arange(1, epochs + 1), history)
    axes.legend(["Training", "Validation"])
    axes.set(xlabel="Epoch", ylabel="Normalized delta MSE")
    figure.savefig(output / "loss.png")
    return load_predictor(output / "model.pt")


def load_predictor(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = make_model()
    model.load_state_dict(checkpoint["weights"])
    model.eval()

    def predict(states, actions):
        inputs = torch.tensor(np.concatenate([states, actions], axis=-1), dtype=torch.float32)
        with torch.no_grad():
            deltas = model((inputs - checkpoint["input_mean"]) / checkpoint["input_scale"])
            result = np.asarray(states) + (deltas * checkpoint["delta_scale"] + checkpoint["delta_mean"]).numpy()
        if not np.isfinite(result).all():
            raise RuntimeError("Nonfinite prediction")
        return result

    return predict


def baseline(states, actions, constant_velocity=False):
    result = states.copy()
    if constant_velocity:
        result[..., :2] += DT * states[..., 2:]
    return result


def errors(predicted, actual):
    difference = predicted - actual
    return {
        "angle_rmse_deg": np.rad2deg(np.sqrt(np.mean(difference[..., :2] ** 2, axis=tuple(range(difference.ndim - 1))))).tolist(),
        "velocity_rmse_rad_s": np.sqrt(np.mean(difference[..., 2:] ** 2, axis=tuple(range(difference.ndim - 1)))).tolist(),
    }


def new_figure():
    figure = Figure(figsize=(8, 4), layout="constrained")
    FigureCanvasAgg(figure)
    return figure, figure.subplots()


def evaluate(dataset, predict, output):
    selected = dataset["split"] == 2
    states, actions = dataset["states"][selected], dataset["actions"][selected]
    predictors = {
        "learned": predict,
        "no_change": baseline,
        "constant_velocity": lambda states, actions: baseline(states, actions, True),
    }
    report = {"one_step": {}, "rollout": {}, "by_scenario": {}, "joint_order": ["shoulder", "elbow"]}
    paths = {}
    for name, predictor in predictors.items():
        predicted = predictor(states[:, :-1], actions)
        report["one_step"][name] = errors(predicted, states[:, 1:])
        report["by_scenario"][name] = {}
        for scenario, label in enumerate(dataset.get("metadata", {}).get("scenarios", SCENARIOS)):
            mask = dataset["scenarios"][selected] == scenario
            if mask.any():
                report["by_scenario"][name][label] = errors(predicted[mask], states[mask, 1:])
        horizon_errors = []
        for horizon in range(1, 11):
            predicted_state = states[:, :states.shape[1] - horizon].copy()
            for offset in range(horizon):
                predicted_state = predictor(predicted_state, actions[:, offset:offset + predicted_state.shape[1]])
            horizon_errors.append(errors(predicted_state, states[:, horizon:]))
        report["rollout"][name] = horizon_errors
        trajectory = [states[0, 0].copy()]
        for action in actions[0, :10]:
            trajectory.append(predictor(trajectory[-1], action))
        paths[name] = np.asarray(trajectory)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    figure, axes = new_figure()
    for name in predictors:
        axes.plot(np.arange(1, 11) * DT, np.mean([entry["angle_rmse_deg"] for entry in report["rollout"][name]], axis=1), label=name)
    axes.set(xlabel="Prediction horizon (s)", ylabel="Mean joint angle RMSE (degrees)")
    axes.legend()
    figure.savefig(output / "rollout_error.png")
    figure = Figure(figsize=(9, 6), layout="constrained")
    FigureCanvasAgg(figure)
    for joint, axes in enumerate(figure.subplots(2, 1)):
        axes.plot(np.arange(11) * DT, np.rad2deg(states[0, :11, joint]), label="MuJoCo", color="black")
        for name, trajectory in paths.items():
            axes.plot(np.arange(11) * DT, np.rad2deg(trajectory[:, joint]), label=name)
        axes.set(xlabel="Time (s)", ylabel=f"{report['joint_order'][joint]} (degrees)")
        axes.legend()
    figure.savefig(output / "trajectory.png")
    print(json.dumps(report["one_step"], indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["collect", "train", "evaluate", "run"])
    parser.add_argument("--dataset", type=Path, default=Path("data/arm_dynamics.npz"))
    parser.add_argument("--output", type=Path, default=Path("runs/arm_dynamics"))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile", choices=["original", "broad"], default="original")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.command in ("collect", "run"):
        dataset = collect(args.dataset, args.episodes, args.steps, args.seed, args.profile)
        print("Episodes per split:", np.bincount(dataset["split"]).tolist())
    else:
        dataset = load_dataset(args.dataset)
    if args.command in ("train", "run"):
        train(dataset, args.output, args.epochs, args.seed)
    if args.command in ("evaluate", "run"):
        checkpoint = torch.load(args.output / "model.pt", map_location="cpu", weights_only=True)
        if checkpoint["metadata"] != dataset["metadata"]:
            raise ValueError("Checkpoint and dataset metadata differ")
        evaluate(dataset, load_predictor(args.output / "model.pt"), args.output)


if __name__ == "__main__":
    main()
