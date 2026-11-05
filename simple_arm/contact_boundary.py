"""Paired near-miss/contact collection and controlled mixed-data experiment."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np
import torch

from .contact_learning import evaluate, load_dataset, load_predictor, physical_errors, rollout, train, train_multistep
from .contact_simulation import ContactSimulation, link_lengths, model_hash
from .contact_varied import command_sequence, tip_position
from .collection_workers import DEFAULT_WORKERS, RNG_SCHEME, collect_tasks, effective_workers, get_simulation, unit_generator


CATEGORIES = ["overhead", "near_miss", "light_contact", "clear_push"]
ROOT = Path("data/arm_cube_contact_boundary")
RUN = Path("runs/arm_cube_contact_boundary")
OLD_MODEL = Path("runs/arm_cube_dynamics_varied_recursive/model.pt")
KEYS = ("states", "actions", "contacts", "starting_contact", "split", "kinds")


def fingerprint(dataset):
    return hashlib.sha256(b"".join(dataset[k].tobytes() for k in ("states", "actions", "split"))).hexdigest()


def save_dataset(path, dataset, summary):
    path = Path(path)
    if path.exists() or path.with_suffix(".json").exists():
        raise FileExistsError(f"Preserving existing data: {path}")
    dataset["metadata"]["data_sha256"] = fingerprint(dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: v for k, v in dataset.items() if k != "metadata"}, metadata=json.dumps(dataset["metadata"]))
    path.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def load_boundary(path):
    dataset = load_dataset(path)
    with np.load(path, allow_pickle=False) as archive:
        for key in ("group_ids", "diagnostics", "family", "source"):
            if key in archive:
                dataset[key] = archive[key].copy()
    if "group_ids" in dataset:
        if dataset["group_ids"].shape != dataset["split"].shape:
            raise ValueError("Invalid group IDs")
        for group in np.unique(dataset["group_ids"]):
            if len(np.unique(dataset["split"][dataset["group_ids"] == group])) != 1:
                raise ValueError("Related group crosses dataset splits")
    return dataset


def simulate(simulation, arm, cube, yaw, velocity, commands, diagnostics=True):
    initial = simulation.initialize(arm, cube, yaw, velocity)
    states = [initial]
    contacts = np.empty((200, 10), dtype=bool)
    starting = np.empty(200, dtype=bool)
    trace = [initial[:2].copy()]
    max_motion, contact_count, overhead = 0., 0, False
    minimum = np.full(3, 1.)
    segment = np.zeros(6)
    geom_ids = [simulation.model.geom(name).id for name in ("upper_link", "lower_link", "pusher")]
    limits = simulation.action_limits
    for step in range(200):
        action = commands[step // 5]
        simulation.data.ctrl[simulation.actuator_ids] = action
        starting[step] = simulation.contact()
        for substep in range(10):
            mujoco.mj_step(simulation.model, simulation.data)
            mujoco.mj_forward(simulation.model, simulation.data)
            q = simulation.data.qpos[simulation.qpos_indices]
            if np.any(q < limits[:, 0] - 1e-4) or np.any(q > limits[:, 1] + 1e-4):
                raise ValueError("Actual joint-limit excursion")
            trace.append(q.copy())
            active = simulation.contact()
            contacts[step, substep] = active
            contact_count += int(active)
            max_motion = max(max_motion, float(np.linalg.norm(simulation.data.xpos[simulation.cube_id, :2] - initial[4:6])))
            if diagnostics:
                for index, geom_id in enumerate(geom_ids):
                    gap = mujoco.mj_geomDistance(simulation.model, simulation.data, geom_id, simulation.cube_geom, 1., segment)
                    minimum[index] = min(minimum[index], gap)
                    if index < 2 and 0.001 <= gap <= 0.004 and np.linalg.norm(segment[3:5] - segment[:2]) <= 0.001:
                        overhead = True
        states.append(simulation.extract())
    # Diagnostics never enter the predictor input or residual target.
    diagnostic = np.r_[max_motion, contact_count * simulation.model.opt.timestep, minimum, float(overhead)]
    return np.asarray(states), contacts, starting, diagnostic, np.asarray(trace)


def classify(diagnostic):
    motion, duration, _, _, pusher_gap, overhead = diagnostic
    return np.array([
        duration == 0 and motion <= 0.002 and overhead > 0 and pusher_gap > 0.005,
        duration == 0 and motion <= 0.002 and 0.0005 <= pusher_gap <= 0.005,
        0 < duration <= 0.2 + 1e-9 and motion <= 0.02,
        duration >= 0.1 - 1e-9 and motion >= 0.05,
    ])


def pusher_gap(path, cube, yaw):
    # Exact horizontal box/capsule separation while cube orientation is fixed.
    # The pusher's vertical segment overlaps the supported cube's height.
    relative = path - cube
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    local = relative @ rotation
    return np.linalg.norm(np.maximum(np.abs(local) - 0.035, 0), axis=1).min() - 0.022


def boundary_offset(path, point, normal, yaw, gap):
    low, high = 0., 0.1
    while high < 0.8 and pusher_gap(path, point + high * normal, yaw) < gap:
        high *= 2
    if pusher_gap(path, point + high * normal, yaw) < gap:
        raise ValueError("No clearance bracket")
    for _ in range(25):
        middle = (low + high) / 2
        if pusher_gap(path, point + middle * normal, yaw) < gap:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def group_split(groups, generator):
    if groups < 40 or groups % 40:
        raise ValueError("Use a multiple of 40 groups for balanced, grouped splits")
    result = np.empty(groups, dtype=int)
    for family in range(4):
        ids = generator.permutation(np.arange(family, groups, 4))
        first = int(len(ids) * 0.7)
        val = int(groups * 0.15) // 4 + int(family < int(groups * 0.15) % 4)
        result[ids[:first]], result[ids[first:first+val]], result[ids[first+val:]] = 0, 1, 2
    return np.repeat(result, 4)


def _collect_group(task):
    seed, group = task
    generator = unit_generator(seed, group, 10)
    simulation = get_simulation()
    lengths, limits = link_lengths(), simulation.action_limits
    rejection = {"group_attempts": 0, "candidate_initialization": 0, "category_mismatch": 0}
    accepted = None
    candidate_attempts = np.zeros(4, dtype=int)
    for attempt in range(100):
        rejection["group_attempts"] += 1
        arm = generator.uniform(limits[:, 0] + 0.2, limits[:, 1] - 0.2)
        velocity = generator.uniform(-0.3, 0.3, 2) if generator.random() < 0.5 else np.zeros(2)
        commands = command_sequence(generator, arm, limits, group % 4)
        # One varied approach followed by a hold exposes release/settling and
        # avoids repeatedly crossing the same candidate contact boundary.
        stop = int(generator.integers(10, 21))
        commands[stop:] = commands[stop - 1]
        yaw = generator.uniform(-np.pi, np.pi)
        try:
            *_, qtrace = simulate(simulation, arm, [0.8, 0.8], yaw, velocity, commands, diagnostics=False)
        except ValueError:
            continue
        trace = tip_position(qtrace, lengths)
        index = int(generator.integers(100, min(900, stop * 50)))
        tangent = trace[index + 10] - trace[index - 10]
        if np.linalg.norm(tangent) < 0.001:
            continue
        normal = generator.choice([-1, 1]) * np.array([-tangent[1], tangent[0]]) / np.linalg.norm(tangent)
        point = trace[index]
        candidates = []
        # Horizontal link passes: sample interior points along either link.
        q = qtrace[index]
        elbow = lengths[0] * np.array([np.cos(q[0]), np.sin(q[0])])
        for fraction in (0.6, 0.8, 0.4):
            candidates.append((0, fraction * elbow))
            candidates.append((0, elbow + fraction * (point - elbow)))
        try:
            for gap in generator.permutation([0.001, 0.0025, 0.004]):
                candidates.append((1, point + boundary_offset(trace, point, normal, yaw, gap) * normal))
            for gap in (-0.00025, -0.00075, -0.002, -0.005, -0.01):
                candidates.append((2, point + boundary_offset(trace, point, normal, yaw, gap) * normal))
            edge = boundary_offset(trace, point, normal, yaw, 0.)
            for factor in (0.5, 0.25, 0.):
                candidates.append((3, point + factor * edge * normal))
        except ValueError:
            continue
        found = {}
        for category, cube in candidates:
            if category in found or np.linalg.norm(cube) < 0.12:
                continue
            if candidate_attempts[category] >= 100:
                raise RuntimeError(f"Group {group}: category {CATEGORIES[category]} exhausted 100 candidate attempts")
            candidate_attempts[category] += 1
            try:
                result = simulate(simulation, arm, cube, yaw, velocity, commands)
            except ValueError:
                rejection["candidate_initialization"] += 1
                continue
            if classify(result[3])[category]:
                found[category] = result
            else:
                rejection["category_mismatch"] += 1
        if len(found) == 4:
            accepted = found
            break
    if accepted is None:
        raise RuntimeError(f"Group {group} failed after 100 attempts: {rejection}")
    return group, {key: value[:4] for key, value in accepted.items()}, np.repeat(commands, 5, axis=0), rejection


def collect(path, groups=320, seed=291, *, workers=DEFAULT_WORKERS):
    path = Path(path)
    if path.exists() or path.with_suffix(".json").exists():
        raise FileExistsError(path)
    effective_workers(workers, groups)
    collection_started = time.perf_counter()
    split = group_split(groups, unit_generator(seed, 0, 11))
    lengths = link_lengths()
    all_states = np.empty((groups * 4, 201, 13))
    all_actions = np.empty((groups * 4, 200, 2))
    all_contacts = np.empty((groups * 4, 200, 10), dtype=bool)
    all_starting = np.empty((groups * 4, 200), dtype=bool)
    all_diagnostics = np.empty((groups * 4, 6))
    rejection = {"group_attempts": 0, "candidate_initialization": 0, "category_mismatch": 0}
    tasks = ((seed, group) for group in range(groups))
    for completed, (group, accepted, commands, rejected) in enumerate(collect_tasks(_collect_group, tasks, workers), 1):
        for key in rejection:
            rejection[key] += rejected[key]
        for category in range(4):
            index = group * 4 + category
            states, contacts, starting, diagnostic = accepted[category]
            all_states[index], all_actions[index] = states, commands
            all_contacts[index], all_starting[index] = contacts, starting
            all_diagnostics[index] = diagnostic
        if completed % 4 == 0:
            print(f"Boundary groups {completed}/{groups}; {rejection}", flush=True)
    collection_seconds = time.perf_counter() - collection_started
    dataset = dict(states=np.asarray(all_states), actions=np.asarray(all_actions), contacts=np.asarray(all_contacts),
                   starting_contact=np.asarray(all_starting), split=split, kinds=np.tile(np.arange(4), groups),
                   group_ids=np.repeat(np.arange(groups), 4), diagnostics=np.asarray(all_diagnostics),
                   family=np.repeat(np.arange(groups) % 4, 4))
    dataset["metadata"] = {"seed": seed, "episodes": groups * 4, "base_dt": 0.02, "physics_dt": 0.002,
                           "model_hash": model_hash(), "lengths": lengths, "collector": "contact_boundary_v2", "rng_scheme": RNG_SCHEME,
                           "kinds": CATEGORIES, "diagnostic_columns": ["max_excursion_m", "contact_duration_s", "upper_gap_m", "lower_gap_m", "pusher_gap_m", "overhead_witness"],
                           "paired_groups": True, "mass_kg": 0.1, "friction": 0.7,
                           "state": "arm q2, arm v2, cube XYZ3, sin_yaw, cos_yaw, cube world v3, yaw_rate",
                           "action": "absolute shoulder/elbow position targets in radians; held 100 ms"}
    summary = {"groups": groups, "episodes": groups * 4, "splits": np.bincount(split).tolist(),
               "category_counts": {name: groups for name in CATEGORIES}, "rejections": rejection,
               "diagnostic_ranges": {name: {"min": dataset["diagnostics"][i::4].min(axis=0).tolist(),
                                               "max": dataset["diagnostics"][i::4].max(axis=0).tolist()} for i, name in enumerate(CATEGORIES)}}
    summary["execution"] = {"workers_requested": workers, "workers_used": effective_workers(workers, groups),
                            "collection_seconds": collection_seconds}
    save_dataset(path, dataset, summary)
    print(json.dumps(summary, indent=2), flush=True)
    return dataset


def merge(focused_path, broad_path, output):
    focused, broad = load_boundary(focused_path), load_dataset(broad_path)
    if focused["metadata"]["model_hash"] != broad["metadata"]["model_hash"]:
        raise ValueError("Physical model mismatch")
    dataset = {k: np.concatenate([broad[k], focused[k]]) for k in KEYS}
    count = len(broad["states"])
    dataset["source"] = np.r_[np.zeros(count, dtype=int), np.ones(len(focused["states"]), dtype=int)]
    dataset["group_ids"] = np.r_[np.arange(count), focused["group_ids"] + count]
    dataset["diagnostics"] = np.concatenate([np.full((count, 6), np.nan), focused["diagnostics"]])
    dataset["family"] = np.r_[broad["kinds"], focused["family"]]
    dataset["metadata"] = dict(focused["metadata"], collector="contact_boundary_mixed_v1", episodes=len(dataset["states"]),
                               parents=[broad["metadata"], focused["metadata"]])
    save_dataset(output, dataset, {"episodes": len(dataset["states"]), "splits": np.bincount(dataset["split"]).tolist(),
                                 "original_splits_preserved": True, "parents": [str(broad_path), str(focused_path)]})
    return dataset


def evaluate_boundary(dataset, checkpoint, output, independent=False):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    provenance = [saved["metadata"], *saved["metadata"].get("parents", [])]
    seeds = [parent["seed"] for parent in provenance]
    if independent:
        if dataset["metadata"]["seed"] in seeds:
            raise ValueError("Independent evaluation overlaps a training collection seed")
        dataset = dict(dataset, split=np.full_like(dataset["split"], 2))
    elif dataset["metadata"]["seed"] in seeds and dataset["metadata"] not in provenance:
        raise ValueError("Evaluation provenance mismatch")
    predict = load_predictor(checkpoint, 5)
    report = evaluate(dataset, predict, output, 5)
    selected = dataset["split"] == 2
    actual = dataset["states"][selected, ::5]
    with np.load(output / "test_rollouts.npz") as archive:
        predictions = {name: archive[name].copy() for name in ("learned", "stationary", "constant_velocity")}
    report["checkpoint"] = str(checkpoint)
    report["category_metrics"] = {}
    if "source" in dataset:
        from .contact_varied import FAMILIES
        masks = {f"{source}_{name}": (dataset["source"][selected] == source_id) & (dataset["kinds"][selected] == index)
                 for source_id, source, names in ((0, "broad", FAMILIES), (1, "boundary", CATEGORIES))
                 for index, name in enumerate(names)}
    else:
        names = dataset["metadata"]["kinds"]
        masks = {name: dataset["kinds"][selected] == index for index, name in enumerate(names)}
    for name, mask in masks.items():
        if not mask.any():
            continue
        group = {}
        for label, trajectory in predictions.items():
            p, a = trajectory[mask], actual[mask]
            drift = np.linalg.norm(p[..., 4:6] - a[:, :1, 4:6], axis=-1).max(axis=1)
            group[label] = {"episodes": int(mask.sum()), "mean_max_motion_m": float(drift.mean()),
                            "median_max_motion_m": float(np.median(drift)), "p95_max_motion_m": float(np.quantile(drift, .95)),
                            "fraction_max_motion_over_2mm": float((drift > .002).mean()),
                            "horizon_endpoints": {str(sec): physical_errors(p[:, sec*10], a[:, sec*10], a[:, 0]) for sec in (1, 2, 4)}}
        group["learned"]["one_step"] = physical_errors(predict(actual[mask, :-1], dataset["actions"][selected, ::5][mask]), actual[mask, 1:], actual[mask, :-1])
        report["category_metrics"][name] = group
    (output / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


def compare_results(root):
    root = Path(root)
    output = root / "comparison.json"
    if output.exists():
        raise FileExistsError(output)
    reports = {name: json.loads((root / name / "evaluation.json").read_text(encoding="utf-8"))
               for name in ("boundary_old", "boundary_firstpass", "boundary_new", "broad_old", "broad_firstpass", "broad_new")}
    result = {"targets": {"mean_max_drift_reduction_fraction": 0.30, "maximum_broad_push_rmse_ratio": 1.10}, "models": {}}
    for name, boundary, broad in (("old", "boundary_old", "broad_old"), ("firstpass", "boundary_firstpass", "broad_firstpass"), ("refined", "boundary_new", "broad_new")):
        groups = reports[boundary]["category_metrics"]
        drift = np.mean([groups[key]["learned"]["mean_max_motion_m"] for key in ("overhead", "near_miss")])
        result["models"][name] = {"mean_max_drift_m": float(drift),
            "broad_endpoint_rmse_m": reports[broad]["endpoint"]["learned"]["displacement_vector_rmse_m"],
            "broad_push_endpoint_rmse_m": reports[broad]["moving_episode_endpoint"]["learned"]["displacement_vector_rmse_m"]}
    old = result["models"]["old"]
    for name in ("firstpass", "refined"):
        metrics = result["models"][name]
        metrics["drift_reduction_fraction"] = 1 - metrics["mean_max_drift_m"] / max(old["mean_max_drift_m"], 1e-12)
        metrics["broad_push_rmse_ratio"] = metrics["broad_push_endpoint_rmse_m"] / max(old["broad_push_endpoint_rmse_m"], 1e-12)
        metrics["passes_drift_target"] = metrics["drift_reduction_fraction"] >= .30
        metrics["passes_pushing_target"] = metrics["broad_push_rmse_ratio"] <= 1.10
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["collect", "merge", "train", "refine", "evaluate", "compare"])
    parser.add_argument("--dataset", type=Path, default=ROOT / "mixed.npz")
    parser.add_argument("--focused", type=Path, default=ROOT / "focused.npz")
    parser.add_argument("--broad", type=Path, default=Path("data/arm_cube_dynamics_varied.npz"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--groups", type=int, default=320)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Collection processes; use 1 for sequential collection")
    parser.add_argument("--seed", type=int, default=291)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--independent", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.command == "compare":
        compare_results(args.output or RUN)
    elif args.command == "collect":
        collect(args.output or args.focused, args.groups, args.seed, workers=args.workers)
    elif args.command == "merge":
        merge(args.focused, args.broad, args.output or args.dataset)
    else:
        dataset = load_boundary(args.dataset)
        output = args.output or RUN / ("firstpass" if args.command == "train" else "recursive" if args.command == "refine" else "evaluation")
        if output.exists():
            raise FileExistsError(output)
        if args.command == "train":
            train(dataset, output, 5, args.epochs or 150)
        elif args.command == "refine":
            train_multistep(dataset, args.checkpoint or RUN / "firstpass/model.pt", output, args.epochs or 80)
        else:
            evaluate_boundary(dataset, args.checkpoint or RUN / "recursive/model.pt", output, args.independent)


if __name__ == "__main__":
    main()
