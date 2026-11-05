"""Actual execution and reproducible, non-overwriting learned-control evaluation."""

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from .contact_boundary import load_boundary
from .contact_learning import new_plot
from .contact_simulation import ContactSimulation
from .contact_viewer import set_display_state
from .cube_mpc import CEMController, DT, ModelAdapter, validate_command_rate


CASES = (23, 99, 139, 159)
SEEDS = (42, 43, 44)
TOLERANCE = .02
SUCCESS_SECONDS = .5


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes(checkpoint, dataset):
    paths = set(Path("data").rglob("*.npz")) | set(Path("runs").rglob("*.pt"))
    paths.update([Path(checkpoint), Path(dataset), Path("models/arm.xml")])
    return {str(p): file_hash(p) for p in sorted(paths)}


def load_cases(path):
    data = load_boundary(path)
    if data["metadata"]["seed"] != 292 or data["metadata"]["kinds"] != ["overhead", "near_miss", "light_contact", "clear_push"]:
        raise ValueError("Expected saved independent boundary dataset (seed 292)")
    cases = {}
    for index in CASES:
        initial = data["states"][index, 0].copy()
        if data["kinds"][index] != 3 or np.linalg.norm(initial[2:4]) > 1e-6 or np.linalg.norm(initial[9:11]) > 1e-6:
            raise ValueError("Expected stationary-start clear push")
        # Recorded commands and contact diagnostics are never returned.
        cases[index] = (initial, data["states"][index, -1, 4:6].copy())
    return cases


def initialize(initial):
    simulation = ContactSimulation()
    set_display_state(simulation, simulation.data, initial)
    simulation.data.ctrl[simulation.actuator_ids] = initial[:2]
    if not np.allclose(simulation.extract(), initial, atol=1e-12, rtol=0):
        raise RuntimeError("Saved initial state was not restored exactly")
    return simulation


def execute_first(simulation, command):
    """One command, exactly 50 physics steps, with observations after stepping."""
    if simulation.frame_skip != 1 or not np.isclose(simulation.model.opt.timestep * 50, DT, atol=1e-12):
        raise ValueError("Execution requires 50 x 2 ms physics steps")
    limits = simulation.action_limits
    command = np.asarray(command)
    if command.shape != (2,) or not np.isfinite(command).all() or np.any(command < limits[:, 0]) or np.any(command > limits[:, 1]):
        raise ValueError("Illegal executed command")
    fine = []
    for _ in range(50):
        simulation.step(command)
        fine.append(simulation.extract())
    return simulation.extract(), np.asarray(fine)


def run_episode(initial, target, predict, config, duration=5., mode="cem"):
    simulation = initialize(initial)
    joint_limits = simulation.model.jnt_range[simulation.joint_ids].copy()
    planner = CEMController(predict, simulation.action_limits, joint_limits, config) if mode == "cem" else None
    previous = initial[:2].copy()
    states, fine_states, commands, predictions, diagnostics, timings, costs = [simulation.extract()], [], [], [], [], [], []
    for step in range(round(duration / DT)):
        tick = time.perf_counter()
        plan = planner.plan(states[-1], target, previous) if planner is not None else None
        elapsed = time.perf_counter() - tick if planner is not None else 0.
        command = plan.actions[0] if plan is not None else initial[:2].copy()
        validate_command_rate(command, previous, config.command_rate_limit)
        observed, fine = execute_first(simulation, command)
        states.append(observed)
        fine_states.extend(fine)
        commands.append(command.copy())
        predictions.append(plan.predicted if plan is not None else np.full((config.horizon, 13), np.nan))
        diagnostic = plan.diagnostics if plan is not None else {"fallback": False, "rejected": 0}
        diagnostics.append(diagnostic)
        timings.append(elapsed)
        costs.append(plan.cost if plan is not None else None)
        if diagnostic["fallback"]:
            print(f"fallback step={step}: all neural candidates infeasible; holding previous command", flush=True)
        previous = command.copy()
    return {"states": np.asarray(states), "fine_states": np.asarray(fine_states), "commands": np.asarray(commands),
            "predictions": np.asarray(predictions), "timings": np.asarray(timings),
            "costs": costs, "diagnostics": diagnostics, "joint_limits": joint_limits}


def sustained_success(distances, physics_dt=.002, tolerance=TOLERANCE, window=SUCCESS_SECONDS):
    count = round(window / physics_dt)
    return bool(len(distances) >= count and np.all(np.asarray(distances)[-count:] <= tolerance))


def timing_metrics(timings):
    return {"mean_s": float(np.mean(timings)), "median_s": float(np.median(timings)),
            "p95_s": float(np.quantile(timings, .95)), "max_s": float(np.max(timings)),
            "deadline_overruns": int(np.count_nonzero(np.asarray(timings) > DT)), "plans": len(timings)}


def metrics(episode, target):
    states, fine, predicted = episode["states"], episode["fine_states"], episode["predictions"]
    distance = np.linalg.norm(states[:, 4:6] - target, axis=-1)
    fine_distance = np.linalg.norm(fine[:, 4:6] - target, axis=-1)
    actual_delta = states[1:, 4:6] - states[:-1, 4:6]
    predicted_delta = predicted[:, 0, 4:6] - states[:-1, 4:6]
    valid = np.isfinite(predicted_delta).all(axis=1)
    suspected = valid & (np.linalg.norm(predicted_delta, axis=1) >= .005) & (np.linalg.norm(actual_delta, axis=1) <= .001)
    limits = episode["joint_limits"]
    excursions = np.maximum(np.maximum(limits[:, 0] - fine[:, :2], fine[:, :2] - limits[:, 1]), 0)
    first_step = []
    for step in range(len(actual_delta)):
        item = {"step": step, "actual_displacement_xy_m": actual_delta[step].tolist(),
                "actual_displacement_m": float(np.linalg.norm(actual_delta[step])),
                "actual_target_progress_m": float(distance[step] - distance[step + 1]),
                "possible_exploitation": bool(suspected[step])}
        if valid[step]:
            item.update(predicted_displacement_xy_m=predicted_delta[step].tolist(),
                        predicted_displacement_m=float(np.linalg.norm(predicted_delta[step])),
                        displacement_error_m=float(np.linalg.norm(predicted_delta[step] - actual_delta[step])),
                        predicted_target_progress_m=float(distance[step] - np.linalg.norm(predicted[step, 0, 4:6] - target)),
                        unverified_terminal_progress_forecast_m=float(distance[step] - np.linalg.norm(predicted[step, -1, 4:6] - target)))
        first_step.append(item)
    return {"initial_distance_m": float(distance[0]), "minimum_distance_m": float(min(distance.min(), fine_distance.min())),
            "final_distance_m": float(distance[-1]), "distance_reduction_m": float(distance[0] - distance[-1]),
            "success": sustained_success(fine_distance), "timing": timing_metrics(episode["timings"]),
            "rejected_candidates": sum(d["rejected"] for d in episode["diagnostics"]),
            "fallbacks": sum(d["fallback"] for d in episode["diagnostics"]),
            "actual_joint_limit_excursion_physics_steps": int(np.any(excursions > 0, axis=1).sum()),
            "maximum_actual_joint_limit_excursion_rad": float(excursions.max()),
            "possible_exploitation_steps": int(suspected.sum()), "first_step_comparisons": first_step}


def aggregate(reports, timings):
    return {"runs": len(reports), "success_rate": float(np.mean([r["success"] for r in reports])),
            **{key: float(np.mean([r[key] for r in reports])) for key in (
                "initial_distance_m", "minimum_distance_m", "final_distance_m", "distance_reduction_m")},
            "possible_exploitation_steps": sum(r["possible_exploitation_steps"] for r in reports),
            "fallbacks": sum(r["fallbacks"] for r in reports),
            "rejected_candidates": sum(r["rejected_candidates"] for r in reports),
            "actual_joint_limit_excursion_physics_steps": sum(r["actual_joint_limit_excursion_physics_steps"] for r in reports),
            "timing": timing_metrics(np.concatenate(timings))}


def plots(output, records, cases):
    for name in ("distance_over_time", "predicted_actual_displacement", "planning_time", "endpoint_comparison"):
        figure, _ = new_plot()
        figure.clear()
        axes = figure.subplots(2, 2).flat
        for panel, case in zip(axes, CASES):
            initial, target = cases[case]
            for key, record in records.items():
                if record["case"] != case:
                    continue
                e = record["episode"]
                label = key.split("_", 1)[1]
                if name == "distance_over_time":
                    panel.plot(np.arange(len(e["states"])) * DT, np.linalg.norm(e["states"][:, 4:6] - target, axis=-1), label=label)
                    panel.set(xlabel="Simulation time (s)", ylabel="Target distance (m)")
                elif name == "predicted_actual_displacement":
                    if record["mode"] != "cem":
                        continue
                    p = e["predictions"][:, 0, 4:6] - e["states"][:-1, 4:6]
                    a = np.diff(e["states"][:, 4:6], axis=0)
                    panel.scatter(np.linalg.norm(p, axis=1) * 1000, np.linalg.norm(a, axis=1) * 1000, s=8, label=label)
                    panel.set(xlabel="Predicted first-step motion (mm)", ylabel="Actual first-step motion (mm)")
                elif name == "planning_time":
                    if record["mode"] == "cem":
                        panel.plot(np.arange(len(e["timings"])) * DT, e["timings"] * 1000, label=label)
                    panel.set(xlabel="Simulation time (s)", ylabel="Planning time (ms)")
                else:
                    panel.plot(e["states"][:, 4], e["states"][:, 5], label=label)
                    panel.scatter(*e["states"][-1, 4:6], marker="x")
                    panel.set(xlabel="Actual X (m)", ylabel="Actual Y (m)", aspect="equal")
            if name == "distance_over_time":
                panel.axhline(TOLERANCE, color="gray", linestyle=":")
            elif name == "planning_time":
                panel.axhline(100, color="red", linestyle=":")
            elif name == "predicted_actual_displacement":
                panel.axvline(5, color="red", linestyle=":")
                panel.axhline(1, color="red", linestyle=":")
            else:
                panel.scatter(*target, marker="*", color="green", s=80, label="target")
                panel.scatter(*initial[4:6], marker="o", color="black", label="start")
            panel.set_title(f"Episode {case}")
            panel.legend(fontsize=6)
        figure.savefig(output / f"{name}.png")


def evaluate(checkpoint, dataset, output, config):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Preserving existing experiment: {output}")
    before = source_hashes(checkpoint, dataset)
    predict = ModelAdapter(checkpoint)
    cases = load_cases(dataset)
    if predict.checkpoint["metadata"]["seed"] == 292 or any(p["seed"] == 292 for p in predict.checkpoint["metadata"].get("parents", [])):
        raise ValueError("Benchmark dataset is not independent of checkpoint")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", {"config": asdict(config), "checkpoint": str(checkpoint), "dataset": str(dataset),
               "cases": list(CASES), "seeds": list(SEEDS), "duration_s": 5, "control_dt_s": DT,
               "physics_steps_per_command": 50, "torch_threads": 2, "tolerance_m": TOLERANCE,
               "success_final_window_s": SUCCESS_SECONDS, "source_hashes_before": before,
               "targets": {str(c): t.tolist() for c, (_, t) in cases.items()},
               "forecast_note": "Predicted terminal progress is an unverified planning forecast; later replans change commands."})
    records, results = {}, {}
    for case, (initial, target) in cases.items():
        for mode, seeds in (("no-action", (None,)), ("cem", SEEDS)):
            for seed in seeds:
                key = f"{case}_{mode}" + (f"_{seed}" if seed is not None else "")
                c = replace(config, seed=seed) if seed is not None else config
                e = run_episode(initial, target, predict, c, mode=mode)
                report = metrics(e, target)
                report.update(case=case, seed=seed, mode=mode)
                np.savez_compressed(output / f"{key}.npz", **{k: e[k] for k in (
                    "states", "fine_states", "commands", "predictions", "timings", "joint_limits")}, target=target)
                write_json(output / f"{key}.json", dict(report, costs=e["costs"], diagnostics=e["diagnostics"]))
                records[key] = {"case": case, "mode": mode, "episode": e}
                results[key] = report
                print(f"{key}: final={report['final_distance_m']:.4f}m success={report['success']} "
                      f"mean_plan={report['timing']['mean_s']:.3f}s overruns={report['timing']['deadline_overruns']}", flush=True)
    controlled = [k for k, r in results.items() if r["mode"] == "cem"]
    summary = {"runs": results, "aggregate": aggregate([results[k] for k in controlled], [records[k]["episode"]["timings"] for k in controlled]),
               "per_case": {}, "per_seed": {}}
    for case in CASES:
        keys = [k for k in controlled if results[k]["case"] == case]
        baseline = results[f"{case}_no-action"]
        for key in keys:
            results[key]["final_distance_improvement_over_baseline_m"] = baseline["final_distance_m"] - results[key]["final_distance_m"]
        summary["per_case"][str(case)] = aggregate([results[k] for k in keys], [records[k]["episode"]["timings"] for k in keys])
        summary["per_case"][str(case)]["baseline_final_distance_m"] = baseline["final_distance_m"]
        summary["per_case"][str(case)]["runs_beating_baseline"] = sum(results[k]["final_distance_improvement_over_baseline_m"] > 0 for k in keys)
    for seed in SEEDS:
        keys = [k for k in controlled if results[k]["seed"] == seed]
        summary["per_seed"][str(seed)] = aggregate([results[k] for k in keys], [records[k]["episode"]["timings"] for k in keys])
    after = {p: file_hash(p) for p in before}
    summary["source_hashes_after"] = after
    summary["sources_unchanged"] = before == after
    write_json(output / "summary.json", summary)
    plots(output, records, cases)
    if before != after:
        raise RuntimeError("Existing dataset/checkpoint hashes changed")
    return summary
