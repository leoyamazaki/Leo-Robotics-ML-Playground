"""Diverse two-joint contact collection, diagnostics, and matched evaluation."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np
import torch

from .contact_learning import (
    BASE_DT, evaluate, labels, load_dataset, load_predictor, physical_errors,
    rollout, train, train_multistep,
)
from .contact_simulation import ContactSimulation, link_lengths, model_hash
from .collection_workers import DEFAULT_WORKERS, RNG_SCHEME, collect_tasks, effective_workers, get_simulation, unit_generator


FAMILIES = ["smooth", "abrupt", "coordinated", "reversals"]
DATASET = Path("data/arm_cube_dynamics_varied.npz")
OUTPUT = Path("runs/arm_cube_dynamics_varied")


def tip_position(q, lengths):
    return np.stack([lengths[0] * np.cos(q[..., 0]) + lengths[1] * np.cos(q.sum(axis=-1)),
                     lengths[0] * np.sin(q[..., 0]) + lengths[1] * np.sin(q.sum(axis=-1))], axis=-1)


def command_sequence(generator, arm, limits, family):
    """Forty legal position commands, with independent joint rates and timings."""
    low, high = limits[:, 0] + 0.15, limits[:, 1] - 0.15
    target = arm.copy()
    direction = generator.choice([-1., 1.], size=2)
    rates = generator.uniform(0.1, 1.5, 2)
    hold = 0
    joint_holds = np.zeros(2, dtype=int)
    joint_pauses = np.zeros(2, dtype=bool)
    commands = []
    for _ in range(40):
        if family == 0:
            for joint in range(2):
                if joint_holds[joint] <= 0:
                    joint_holds[joint] = generator.integers(2, 9)
                    direction[joint] = generator.choice([-1., 1.])
                    rates[joint] = generator.uniform(0.1, 1.5)
                    joint_pauses[joint] = generator.random() < 0.12
            target = target + 0.1 * rates * direction * ~joint_pauses
            crossing = (target < low) | (target > high)
            direction[crossing] *= -1
            target = np.clip(target, low, high)
            joint_holds -= 1
            commands.append(target.copy())
            continue
        if hold <= 0:
            hold = int(generator.integers(1 if family == 1 else 2, 9))
            rates = generator.uniform(0.1, 1.5, 2)
            if family == 1:
                target = np.clip(arm + generator.uniform(-1.2, 1.2, 2), low, high)
            elif family == 2:
                direction = generator.choice([-1., 1.]) * np.array([1., generator.choice([-1., 1.])])
                rates[1] = np.clip(rates[0] * generator.uniform(0.5, 1.5), 0.1, 1.5)
            else:
                direction *= -1
            paused = generator.random() < 0.12
        if family != 1 and not paused:
            target = target + 0.1 * rates * direction
            crossing = (target < low) | (target > high)
            direction[crossing] *= -1
            target = np.clip(target, low, high)
        commands.append(target.copy())
        hold -= 1
    return np.asarray(commands)


def summarize(dataset):
    states, actions = dataset["states"], dataset["actions"][:, ::5]
    contact = dataset["contacts"].reshape(len(states), 40, 50).any(axis=-1)
    excursion = np.rad2deg(np.ptp(states[..., :2], axis=1))
    motion = np.linalg.norm(states[:, -1, 4:6] - states[:, 0, 4:6], axis=-1)

    def distribution(array):
        return {"min": np.min(array, axis=0).tolist(), "median": np.median(array, axis=0).tolist(),
                "p95": np.quantile(array, 0.95, axis=0).tolist(), "max": np.max(array, axis=0).tolist()}

    limits = ContactSimulation().model.jnt_range[:2]
    violations = (states[..., :2] < limits[:, 0] - 1e-4) | (states[..., :2] > limits[:, 1] + 1e-4)
    summary = {"episodes": len(states), "transitions_100ms": len(states) * 40,
               "split_episodes": np.bincount(dataset["split"], minlength=3).tolist(),
               "both_joints_over_15deg_fraction": float(np.all(excursion >= 15, axis=-1).mean()),
               "contact_window_fraction": float(contact.mean()), "moving_over_2cm_fraction": float((motion > 0.02).mean()),
               "joint_excursion_deg": distribution(excursion),
               "joint_positions_rad": distribution(states[..., :2].reshape(-1, 2)),
               "initial_joint_positions_rad": distribution(states[:, 0, :2]),
               "both_joints_moving_fraction_100ms": float(np.all(np.abs(np.diff(states[:, ::5, :2], axis=1)) > 0.005, axis=-1).mean()),
               "absolute_joint_velocity_rad_s": distribution(np.abs(states[..., 2:4]).reshape(-1, 2)),
               "command_jump_rad": distribution(np.abs(np.diff(actions, axis=1)).reshape(-1, 2)),
               "initial_cube_xy_m": distribution(states[:, 0, 4:6]),
               "initial_cube_yaw_rad": distribution(np.arctan2(states[:, 0, 7], states[:, 0, 8])),
               "initial_cube_velocity": distribution(states[:, 0][:, [9, 10, 12]]),
               "joint_limit_violation_samples": int(violations.any(axis=-1).sum()),
               "contact_categories_100ms": {k: int(v.sum()) for k, v in labels(dataset, 5).items()},
               "families": {}}
    for kind, name in enumerate(FAMILIES):
        selected = dataset["kinds"] == kind
        summary["families"][name] = {"episodes": int(selected.sum()),
            "joint_excursion_deg": distribution(excursion[selected]),
            "contact_window_fraction": float(contact[selected].mean()),
            "moving_over_2cm_fraction": float((motion[selected] > 0.02).mean())}
    return summary


def check_coverage(summary):
    failures = []
    for key, threshold in (("both_joints_over_15deg_fraction", 0.75), ("contact_window_fraction", 0.15),
                           ("moving_over_2cm_fraction", 0.5)):
        if summary[key] < threshold:
            failures.append(f"{key}={summary[key]:.3f} < {threshold}")
    if summary["joint_limit_violation_samples"]:
        failures.append("Actual joint-limit excursions detected")
    if failures:
        raise RuntimeError("Dataset coverage failed: " + "; ".join(failures))


def _collect_varied_episode(task):
    seed, episode = task
    generator = unit_generator(seed, episode, 20)
    simulation = get_simulation()
    lengths, limits = link_lengths(), simulation.action_limits
    family = episode % len(FAMILIES)
    states = np.empty((1, 201, 13))
    actions = np.empty((1, 200, 2))
    contacts = np.empty((1, 200, 10), dtype=bool)
    starting = np.empty((1, 200), dtype=bool)
    specs, rejected = [], {"initialization": 0, "joint_limits": 0, "motion": 0, "contact": 0}
    # Exact 70/30 targeted/workspace and 50/50 rest/moving mix within each family.
    slot = episode // 4
    targeted = slot % 10 < 7
    moving = slot % 2 == 1
    for attempt in range(50):
        arm = generator.uniform(limits[:, 0] + 0.15, limits[:, 1] - 0.15)
        commands = command_sequence(generator, arm, limits, family)
        yaw = generator.uniform(-np.pi, np.pi)
        arm_velocity = generator.uniform(-0.5, 0.5, 2) if generator.random() < 0.7 else np.zeros(2)
        arm_velocity = np.where((arm < limits[:, 0] + 0.3) | (arm > limits[:, 1] - 0.3), 0, arm_velocity)
        angle = generator.uniform(-np.pi, np.pi)
        speed = generator.uniform(0.02, 0.15) if moving else 0
        cube_velocity = np.array([speed * np.cos(angle), speed * np.sin(angle), generator.uniform(-1, 1) if moving else 0])
        if targeted:
            waypoint = int(generator.integers(1, 8))
            cube = tip_position(commands[waypoint], lengths) + generator.uniform(-0.015, 0.015, 2)
        else:
            random_pose = generator.uniform(limits[:, 0] + 0.15, limits[:, 1] - 0.15)
            cube = tip_position(random_pose, lengths)
        if np.linalg.norm(cube) < 0.12:
            rejected["initialization"] += 1
            continue
        try:
            initial = simulation.initialize(arm, cube, yaw, arm_velocity, cube_velocity)
        except ValueError:
            rejected["initialization"] += 1
            continue
        states[0, 0] = initial
        joint_violation = False
        for step in range(200):
            action = commands[step // 5]
            actions[0, step] = action
            starting[0, step] = simulation.contact()
            simulation.data.ctrl[simulation.actuator_ids] = action
            for substep in range(10):
                mujoco.mj_step(simulation.model, simulation.data)
                mujoco.mj_forward(simulation.model, simulation.data)
                contacts[0, step, substep] = simulation.contact()
                q = simulation.data.qpos[simulation.qpos_indices]
                joint_violation |= bool(np.any(q < limits[:, 0] - 1e-4) or np.any(q > limits[:, 1] + 1e-4))
            states[0, step + 1] = simulation.extract()
        trajectory = states[0]
        # Reject impacts with hard stops rather than training outside this experiment's regime.
        if joint_violation:
            rejected["joint_limits"] += 1
            continue
        if not np.all(np.ptp(trajectory[:, :2], axis=0) >= np.deg2rad(15)):
            rejected["motion"] += 1
            continue
        contact_fraction = contacts[0].reshape(40, 50).any(axis=-1).mean()
        displacement = np.linalg.norm(trajectory[-1, 4:6] - trajectory[0, 4:6])
        if targeted and (contact_fraction < 0.225 or displacement <= 0.02):
            rejected["contact"] += 1
            continue
        specs.append({"targeted": targeted, "moving_start": moving, "attempts": attempt + 1,
                      "initial_arm": arm.tolist(), "initial_cube_xy": cube.tolist(), "initial_cube_yaw": yaw,
                      "initial_arm_velocity": arm_velocity.tolist(), "initial_cube_velocity": cube_velocity.tolist()})
        break
    else:
        raise RuntimeError(f"No valid episode {episode} ({FAMILIES[family]}) after 50 attempts: {rejected}")
    return episode, states[0], actions[0], contacts[0], starting[0], specs[0], rejected


def collect_varied(path, episodes=1200, seed=191, *, workers=DEFAULT_WORKERS):
    if episodes < 40 or episodes % 40:
        raise ValueError("Use a multiple of 40 episodes for balanced families and 70/15/15 splits")
    path = Path(path)
    if path.exists() or path.with_suffix(".json").exists():
        raise FileExistsError(f"Preserving existing dataset: {path}")
    effective_workers(workers, episodes)
    collection_started = time.perf_counter()
    generator = unit_generator(seed, 0, 21)
    lengths = link_lengths()
    states = np.empty((episodes, 201, 13))
    actions = np.empty((episodes, 200, 2))
    contacts = np.empty((episodes, 200, 10), dtype=bool)
    starting = np.empty((episodes, 200), dtype=bool)
    kinds = np.arange(episodes) % len(FAMILIES)
    specs = [None] * episodes
    rejected = {"initialization": 0, "joint_limits": 0, "motion": 0, "contact": 0}
    tasks = ((seed, episode) for episode in range(episodes))
    for completed, result in enumerate(collect_tasks(_collect_varied_episode, tasks, workers), 1):
        episode, state, action, contact, start, spec, rejection = result
        states[episode], actions[episode], contacts[episode], starting[episode] = state, action, contact, start
        specs[episode] = spec
        for key in rejected:
            rejected[key] += rejection[key]
        if completed % 20 == 0:
            print(f"Varied collection {completed}/{episodes}; rejections={rejected}", flush=True)
    collection_seconds = time.perf_counter() - collection_started
    split = np.empty(episodes, dtype=int)
    per_family = episodes // 4
    validation_total = int(episodes * 0.15)
    validation_base, validation_extra = divmod(validation_total, 4)
    for kind in range(4):
        indices = generator.permutation(np.flatnonzero(kinds == kind))
        first = int(per_family * 0.7)
        second = first + validation_base + int(kind < validation_extra)
        split[indices[:first]], split[indices[first:second]], split[indices[second:]] = 0, 1, 2
    metadata = {"base_dt": BASE_DT, "seed": seed, "episodes": episodes, "model_hash": model_hash(),
                "lengths": lengths, "kinds": FAMILIES, "collector": "varied_v3", "rng_scheme": RNG_SCHEME,
                "data_sha256": hashlib.sha256(states.tobytes() + actions.tobytes() + split.tobytes()).hexdigest(),
                "state": "arm q2, arm v2, cube XYZ3, sin_yaw, cos_yaw, cube world v3, yaw_rate",
                "action": "absolute shoulder/elbow position targets in radians; held 100 ms",
                "physics_dt": 0.002, "mass_kg": 0.1, "friction": 0.7,
                "joint_margin_rad": 0.15, "ramp_rate_rad_s": [0.1, 1.5],
                "initial_planar_cube_speed_m_s": [0, 0.15], "initial_cube_yaw_rate_rad_s": [-1, 1],
                "selection": "both joints >=15deg; targeted contact >=22.5% windows and displacement >2cm; no hard-stop excursions"}
    dataset = dict(states=states, actions=actions, contacts=contacts, starting_contact=starting, split=split, kinds=kinds, metadata=metadata)
    summary = summarize(dataset)
    check_coverage(summary)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: v for k, v in dataset.items() if k != "metadata"}, metadata=json.dumps(metadata))
    summary.update(metadata=metadata, rejection_counts=rejected, sampling=specs,
                   execution={"workers_requested": workers, "workers_used": effective_workers(workers, episodes),
                              "collection_seconds": collection_seconds})
    path.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("sampling", "metadata")}, indent=2), flush=True)
    return dataset


def evaluate_varied(dataset, checkpoint, output, all_episodes=False):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Preserving existing evaluation: {output}")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["stride"] != 5:
        raise ValueError("Varied experiment uses 100 ms prediction")
    if all_episodes:
        if saved["metadata"]["seed"] == dataset["metadata"]["seed"]:
            raise ValueError("Independent evaluation requires a different seed")
        dataset = dict(dataset, split=np.full_like(dataset["split"], 2))
    elif saved["metadata"]["seed"] == dataset["metadata"]["seed"] and saved["metadata"] != dataset["metadata"]:
        raise ValueError("Same-seed evaluation requires identical dataset provenance")
    predict = load_predictor(checkpoint, 5)
    report = evaluate(dataset, predict, output, 5)
    selected = dataset["split"] == 2
    states = dataset["states"][selected, ::5]
    actions = dataset["actions"][selected, ::5]
    predicted = rollout(predict, states[:, 0], actions)
    with np.load(output / "test_rollouts.npz") as archive:
        trajectories = {name: archive[name].copy() for name in ("learned", "stationary", "constant_velocity")}
    report["checkpoint"] = str(checkpoint)
    report["groups"] = {}
    groups = {name: dataset["kinds"][selected] == index for index, name in enumerate(FAMILIES)}
    initial_speed = np.linalg.norm(states[:, 0, 9:11], axis=-1)
    groups.update(cube_starts_at_rest=initial_speed < 1e-6, cube_starts_moving=initial_speed >= 1e-6)
    for name, mask in groups.items():
        if not mask.any():
            continue
        report["groups"][name] = {"episodes": int(mask.sum()),
            "endpoint": physical_errors(predicted[mask, -1], states[mask, -1], states[mask, 0]),
            "one_step": physical_errors(predict(states[mask, :-1], actions[mask]), states[mask, 1:], states[mask, :-1]),
            "endpoint_comparison": {key: physical_errors(value[mask, -1], states[mask, -1], states[mask, 0]) for key, value in trajectories.items()}}
    (output / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["collect", "train", "refine", "evaluate"])
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--episodes", type=int, default=1200)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Collection processes; use 1 for sequential collection")
    parser.add_argument("--seed", type=int, default=191)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--all-episodes", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.command == "collect":
        collect_varied(args.dataset, args.episodes, args.seed, workers=args.workers)
        return
    dataset = load_dataset(args.dataset)
    if dataset["metadata"].get("collector") not in ("varied_v2", "varied_v3"):
        raise ValueError("This command requires a varied dataset; use contact_learning for older data")
    if args.command in ("train", "refine") and args.output.exists():
        raise FileExistsError(f"Use a new output directory: {args.output}")
    if args.command == "train":
        train(dataset, args.output, 5, args.epochs or 150)
    elif args.command == "refine":
        train_multistep(dataset, args.checkpoint or OUTPUT / "model.pt", args.output, args.epochs or 80)
    else:
        evaluate_varied(dataset, args.checkpoint or OUTPUT / "model.pt", args.output, args.all_episodes)


if __name__ == "__main__":
    main()
