"""Audit an existing checkpoint on unseen, five-second open-loop sequences."""

import argparse
import hashlib
import json
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import mujoco
import numpy as np
import torch

from .arm_learning import DT, errors, load_dataset, load_predictor, state
from .simulation import ArmSimulation, MODEL_PATH


GROUPS = ["irregular_in_range", "rapid_reversals_in_range", "limit_holds", "limit_reversals"]


def autoregressive(predict, initial_state, actions):
    """Only the initial state and future actions are available to this loop."""
    trajectory = [np.asarray(initial_state).copy()]
    for action in actions:
        trajectory.append(predict(trajectory[-1], action))
    return np.asarray(trajectory)


def make_cases(limits, count=12, seed=2026):
    generator = np.random.default_rng(seed)
    cases = []
    for group in GROUPS:
        for _ in range(count):
            initial = generator.uniform([-0.9, -1.2], [0.9, 1.2])
            commands = []
            signs = generator.choice([-1, 1], size=2)
            if group.startswith("limit"):
                initial = signs * limits[:, 1] * generator.uniform(0.94, 0.99, 2)
            while len(commands) < 50:
                if group == "irregular_in_range":
                    target = generator.uniform([-1.2, -1.6], [1.2, 1.6])
                    hold = int(generator.integers(1, 8))
                elif group == "rapid_reversals_in_range":
                    signs *= -1
                    target = signs * generator.uniform([0.75, 1.0], [1.2, 1.6])
                    hold = int(generator.integers(1, 4))
                else:
                    signs *= -1
                    target = np.where(signs > 0, limits[:, 1], limits[:, 0])
                    if group == "limit_holds":
                        hold = int(generator.integers(7, 13))
                    else:
                        hold = int(generator.integers(1, 5))
                commands.extend([target.copy() for _ in range(hold)])
            cases.append((group, initial, np.asarray(commands[:50])))
    return cases


def simulate(initial, actions):
    simulation = ArmSimulation(frame_skip=1)
    cube_address = simulation.model.jnt_qposadr[simulation.model.joint("cube_free").id]
    simulation.data.qpos[cube_address:cube_address + 3] = [1.2, 1.2, 0.12]
    simulation.data.qpos[simulation.qpos_indices] = initial
    simulation.data.ctrl[:] = initial
    mujoco.mj_forward(simulation.model, simulation.data)
    trajectory = [state(simulation)]
    limit_contact_steps = 0
    for action in actions:
        for _ in range(50):
            simulation.step(action)
            limit_contact_steps += int(any(
                constraint == mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT
                for constraint in simulation.data.efc_type
            ))
        trajectory.append(state(simulation))
    return np.asarray(trajectory), limit_contact_steps


def summarize(predicted, actual, actions, limits):
    difference = predicted[:, 1:] - actual[:, 1:]
    near_limit = np.any(
        (actual[:, 1:, :2] < limits[:, 0] + 0.1)
        | (actual[:, 1:, :2] > limits[:, 1] - 0.1), axis=-1,
    )
    reversal = np.zeros(actions.shape[:2], dtype=bool)
    reversal[:, 1:] = np.any(
        (actions[:, 1:] * actions[:, :-1] < 0)
        & (np.abs(actions[:, 1:] - actions[:, :-1]) > 0.5), axis=-1,
    )
    result = {
        "endpoint_errors": {str(seconds): errors(predicted[:, seconds * 10], actual[:, seconds * 10]) for seconds in (1, 3, 5)},
        "trajectory_errors": errors(predicted[:, 1:], actual[:, 1:]),
        "peak_angle_error_deg": np.rad2deg(np.max(np.abs(difference[..., :2]), axis=(0, 1))).tolist(),
        "peak_velocity_error_rad_s": np.max(np.abs(difference[..., 2:]), axis=(0, 1)).tolist(),
    }
    for name, trajectory in (("predicted", predicted), ("actual", actual)):
        overshoot = np.maximum(np.maximum(limits[:, 0] - trajectory[:, 1:, :2], trajectory[:, 1:, :2] - limits[:, 1]), 0)
        result[name + "_limit_exceedance_samples"] = int(np.count_nonzero(np.any(overshoot > 0.001, axis=-1)))
        result[name + "_max_limit_exceedance_deg"] = np.rad2deg(overshoot.max(axis=(0, 1))).tolist()
    for label, mask in (("near_limit", near_limit), ("first_step_after_reversal", reversal)):
        result[label] = {"samples": int(mask.sum()), "errors": errors(predicted[:, 1:][mask], actual[:, 1:][mask]) if mask.any() else None}
    return result


def run(checkpoint_path, dataset_path, output, count=12, seed=2026):
    if count < 1:
        raise ValueError("cases per group must be positive")
    checkpoint_path, output = Path(checkpoint_path), Path(output)
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    dataset = load_dataset(dataset_path)
    if checkpoint["metadata"] != dataset["metadata"]:
        raise ValueError("Training dataset and checkpoint differ")
    if checkpoint["metadata"]["model_sha256"] != hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest():
        raise ValueError("Current physics model differs from checkpoint model")
    limits = ArmSimulation().action_limits
    cases = make_cases(limits, count, seed)
    predict = load_predictor(checkpoint_path)
    actual, predicted, contacts = [], [], []
    for _, initial, actions in cases:
        truth, contact_steps = simulate(initial, actions)
        actual.append(truth)
        predicted.append(autoregressive(predict, truth[0], actions))
        contacts.append(contact_steps)
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    actions = np.asarray([case[2] for case in cases])
    groups = np.asarray([case[0] for case in cases])
    train_states = dataset["states"][dataset["split"] == 0, :-1]
    train_actions = dataset["actions"][dataset["split"] == 0]
    train_inputs = np.concatenate([train_states, train_actions], axis=-1).reshape(-1, 6)
    report = {
        "checkpoint_sha256": checkpoint_hash, "seed": seed, "dt": DT,
        "duration_seconds": 5, "cases_per_group": count,
        "joint_order": ["shoulder", "elbow"],
        "method": "50 predictions from one measured initial state; recorded actions only; no state corrections or clipping",
        "training_input_min": train_inputs.min(axis=0).tolist(),
        "training_input_max": train_inputs.max(axis=0).tolist(), "groups": {},
    }
    output.mkdir(parents=True, exist_ok=True)
    for group in GROUPS:
        mask = groups == group
        metrics = summarize(predicted[mask], actual[mask], actions[mask], limits)
        model_inputs = np.concatenate([predicted[mask, :-1], actions[mask]], axis=-1)
        metrics["predicted_input_outside_training_range_fraction"] = float(np.mean(np.any(
            (model_inputs < train_inputs.min(axis=0)) | (model_inputs > train_inputs.max(axis=0)), axis=-1,
        )))
        metrics["joint_limit_active_physics_steps"] = int(np.asarray(contacts)[mask].sum())
        report["groups"][group] = metrics
        group_indices = np.flatnonzero(mask)
        worst = group_indices[np.argmax(np.mean((predicted[mask, 1:, :2] - actual[mask, 1:, :2]) ** 2, axis=(1, 2)))]
        figure = Figure(figsize=(10, 7), layout="constrained")
        FigureCanvasAgg(figure)
        for joint, axes in enumerate(figure.subplots(2, 1)):
            axes.plot(np.arange(51) * DT, np.rad2deg(actual[worst, :, joint]), label="MuJoCo")
            axes.plot(np.arange(51) * DT, np.rad2deg(predicted[worst, :, joint]), label="Autoregressive prediction")
            axes.step(np.arange(50) * DT, np.rad2deg(actions[worst, :, joint]), where="post", alpha=0.5, label="Position target")
            for limit in limits[joint]:
                axes.axhline(np.rad2deg(limit), color="gray", linestyle=":")
            axes.set(xlabel="Time (s)", ylabel=f"{report['joint_order'][joint]} (degrees)")
            axes.legend()
        figure.suptitle(f"{group}: worst whole-trajectory angle RMSE case")
        figure.savefig(output / f"{group}.png")
    figure = Figure(figsize=(9, 5), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    for group in GROUPS:
        difference = predicted[groups == group, 1:, :2] - actual[groups == group, 1:, :2]
        axes.plot(np.arange(1, 51) * DT, np.rad2deg(np.sqrt(np.mean(difference ** 2, axis=(0, 2)))), label=group)
    axes.set(xlabel="Open-loop prediction horizon (s)", ylabel="Pooled joint angle RMSE (degrees)")
    axes.legend()
    figure.savefig(output / "horizon_errors.png")
    np.savez_compressed(output / "trajectories.npz", actual=actual, predicted=predicted, actions=actions, groups=groups)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    if hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() != checkpoint_hash:
        raise RuntimeError("Checkpoint changed during evaluation")
    for group, metrics in report["groups"].items():
        print(group, json.dumps(metrics["endpoint_errors"]))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/arm_dynamics/model.pt"))
    parser.add_argument("--dataset", type=Path, default=Path("data/arm_dynamics.npz"))
    parser.add_argument("--output", type=Path, default=Path("runs/rollout_stress"))
    parser.add_argument("--cases-per-group", type=int, default=12)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    torch.set_num_threads(2)
    run(args.checkpoint, args.dataset, args.output, args.cases_per_group, args.seed)


if __name__ == "__main__":
    main()
