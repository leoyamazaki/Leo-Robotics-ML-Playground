"""Random-shooting learned MPC and a direct position-control comparison."""

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import mujoco
import numpy as np
import torch

from .arm_learning import DT, STATE_NAMES, load_predictor, state
from .simulation import ArmSimulation, MODEL_PATH


@dataclass(frozen=True)
class Config:
    horizon: int = 8
    candidates: int = 256
    seed: int = 42
    position_weight: float = 1.0
    velocity_weight: float = 0.03
    change_weight: float = 0.08
    terminal_position_weight: float = 8.0
    terminal_velocity_weight: float = 0.1
    command_margin: float = 0.08

    def __post_init__(self):
        if not isinstance(self.horizon, int) or not isinstance(self.candidates, int) or self.horizon < 1 or self.candidates < 4:
            raise ValueError("Require a positive horizon and at least four candidates")
        weights = list(asdict(self).values())[3:]
        if not np.isfinite(weights).all() or min(weights) < 0:
            raise ValueError("Weights and margin must be finite and nonnegative")


def load_dynamics(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    metadata = checkpoint["metadata"]
    if metadata["state_order"] != STATE_NAMES or metadata["dt"] != DT:
        raise ValueError("Checkpoint state conventions or timestep differ")
    if metadata["model_sha256"] != hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest():
        raise ValueError("Physics model differs from training")
    for name, width in (("input_mean", 6), ("input_scale", 6), ("delta_mean", 4), ("delta_scale", 4)):
        value = checkpoint[name]
        if value.shape != (width,) or not torch.isfinite(value).all():
            raise ValueError(f"Invalid checkpoint normalization: {name}")
        if name.endswith("scale") and not (value > 0).all():
            raise ValueError("Normalization scales must be positive")
    return load_predictor(checkpoint_path)


def rollout(predict, initial, actions):
    current = np.broadcast_to(initial, (len(actions), 4)).copy()
    predicted = []
    for offset in range(actions.shape[1]):
        current = predict(current, actions[:, offset])
        predicted.append(current.copy())
    return np.stack(predicted, axis=1)


def trajectory_cost(predicted, actions, goal, previous, joint_limits, config):
    position = np.sum((predicted[..., :2] - goal) ** 2, axis=-1)
    velocity = np.sum(predicted[..., 2:] ** 2, axis=-1)
    changes = np.diff(np.concatenate([np.broadcast_to(previous, (len(actions), 1, 2)), actions], axis=1), axis=1)
    violations = np.maximum(np.maximum(joint_limits[:, 0] - predicted[..., :2], predicted[..., :2] - joint_limits[:, 1]), 0)
    cost = (
        config.position_weight * position.sum(axis=1)
        + config.velocity_weight * velocity.sum(axis=1)
        + config.change_weight * np.sum(changes ** 2, axis=(1, 2))
        + config.terminal_position_weight * position[:, -1]
        + config.terminal_velocity_weight * velocity[:, -1]
    )
    invalid = ~np.isfinite(predicted).all(axis=(1, 2)) | np.any(violations > 0, axis=(1, 2))
    return np.where(invalid, np.inf, cost)


class LearnedMPC:
    def __init__(self, predict, action_limits, joint_limits, config=Config()):
        self.predict = predict
        self.config = config
        self.joint_limits = np.asarray(joint_limits).copy()
        self.bounds = np.column_stack([
            np.maximum(action_limits[:, 0], joint_limits[:, 0] + config.command_margin),
            np.minimum(action_limits[:, 1], joint_limits[:, 1] - config.command_margin),
        ])
        if np.any(self.bounds[:, 0] >= self.bounds[:, 1]):
            raise ValueError("Command margin leaves no feasible commands")
        self.generator = np.random.default_rng(config.seed)
        self.best = None

    def candidates(self, goal, previous):
        count, horizon = self.config.candidates, self.config.horizon
        center = np.clip(goal, self.bounds[:, 0], self.bounds[:, 1])
        actions = np.empty((count, horizon, 2))
        actions[0] = center
        actions[1] = previous
        warm = np.tile(center, (horizon, 1)) if self.best is None else np.vstack([self.best[1:], self.best[-1]])
        actions[2] = warm
        for candidate in range(3, count):
            if candidate < count // 4:
                actions[candidate] = self.generator.uniform(self.bounds[:, 0], self.bounds[:, 1])
            else:
                base = warm if candidate % 2 else np.tile(center, (horizon, 1))
                noise = self.generator.normal(0, 0.35, (horizon, 2))
                for offset in range(1, horizon):
                    noise[offset] = 0.65 * noise[offset - 1] + 0.35 * noise[offset]
                actions[candidate] = base + noise
        return np.clip(actions, self.bounds[:, 0], self.bounds[:, 1])

    def act(self, observed, goal, previous):
        actions = self.candidates(goal, previous)
        predicted = rollout(self.predict, observed, actions)
        costs = trajectory_cost(predicted, actions, goal, previous, self.joint_limits, self.config)
        feasible = np.isfinite(costs)
        if not feasible.any():
            self.best = None
            command = np.clip(previous, self.bounds[:, 0], self.bounds[:, 1])
            return command, {"fallback": True, "selected": -1, "rejected": len(actions), "predicted_next": self.predict(observed, command), "cost": None}
        selected = int(np.argmin(costs))
        self.best = actions[selected].copy()
        return self.best[0].copy(), {
            "fallback": False, "selected": selected, "rejected": int((~feasible).sum()),
            "predicted_next": predicted[selected, 0], "cost": float(costs[selected]),
        }


def scenarios():
    return {
        "short": {"initial": [0, 0], "duration": 4, "targets": [(0, [0.2, -0.3])]},
        "large": {"initial": [-1.0, 1.3], "duration": 5, "targets": [(0, [1.1, -1.5])]},
        "moving_target": {"initial": [0, 0], "duration": 6, "targets": [(0, [1.0, -1.2]), (0.4, [-0.6, 0.8]), (2, [0.5, 0.4])]},
        "near_limits": {"initial": [-0.9, -1.3], "duration": 5, "targets": [(0, [1.4, 1.9])]},
        "reversals": {"initial": [0, 0], "duration": 6, "targets": [(0, [1, -1.3]), (0.6, [-1, 1.3]), (1.2, [1, -1.3]), (2, [-0.5, 0.6])]},
    }


def initialize_simulation(initial):
    simulation = ArmSimulation(frame_skip=1)
    simulation.data.qpos[simulation.qpos_indices] = initial
    cube_address = simulation.model.jnt_qposadr[simulation.model.joint("cube_free").id]
    simulation.data.qpos[cube_address:cube_address + 3] = [1.2, 1.2, 0.12]
    simulation.data.ctrl[:] = initial
    mujoco.mj_forward(simulation.model, simulation.data)
    return simulation


def run_episode(scenario, controller, predict, config, viewer=False):
    simulation = initialize_simulation(scenario["initial"])
    joint_limits = simulation.model.jnt_range[simulation.joint_ids].copy()
    previous = np.asarray(scenario["initial"], dtype=float)
    planner = LearnedMPC(predict, simulation.action_limits, joint_limits, config) if controller == "mpc" else None
    observed_states, goals, commands, timings, diagnostics = [state(simulation)], [], [], [], []
    fine_states, fine_goals = [], []
    active_limits = 0
    context = nullcontext(None)
    if viewer:
        import mujoco.viewer as viewer_module
        context = viewer_module.launch_passive(simulation.model, simulation.data)
    with context as handle:
        if handle is not None:
            with handle.lock():
                handle.cam.lookat[:] = [0.25, 0, 0.05]
                handle.cam.distance = 1.4
        for step_index in range(round(scenario["duration"] / DT)):
            if handle is not None and not handle.is_running():
                break
            tick = time.perf_counter()
            goal = np.asarray([value for start, value in scenario["targets"] if start <= step_index * DT + 1e-9][-1])
            if planner is None:
                command = np.clip(goal, simulation.action_limits[:, 0], simulation.action_limits[:, 1])
                diagnostic = {"fallback": False, "selected": None, "rejected": 0, "cost": None}
            else:
                command, diagnostic = planner.act(observed_states[-1], goal, previous)
            timings.append(time.perf_counter() - tick)
            if not np.isfinite(command).all() or np.any(command < simulation.action_limits[:, 0]) or np.any(command > simulation.action_limits[:, 1]):
                raise RuntimeError("Controller produced an illegal command")
            with handle.lock() if handle is not None else nullcontext():
                for _ in range(50):
                    simulation.step(command)
                    fine_states.append(state(simulation))
                    fine_goals.append(goal.copy())
                    active_limits += int(np.any(simulation.data.efc_type == mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT))
                observed_states.append(state(simulation))
            diagnostics.append(diagnostic)
            goals.append(goal)
            commands.append(command)
            previous = command.copy()
            if handle is not None:
                handle.sync()
                time.sleep(max(0, DT - (time.perf_counter() - tick)))
    if not commands:
        raise RuntimeError("Episode ended before any actions executed")
    return {
        "states": np.asarray(observed_states), "goals": np.asarray(goals),
        "commands": np.asarray(commands), "timings": np.asarray(timings),
        "fine_states": np.asarray(fine_states), "fine_goals": np.asarray(fine_goals),
        "diagnostics": diagnostics, "active_limit_physics_steps": active_limits,
    }


def metrics(episode, scenario, joint_limits):
    actual, goals = episode["fine_states"], episode["fine_goals"]
    difference = actual[:, :2] - goals
    segments = []
    overshoot = np.zeros(2)
    for segment_index, (start, goal) in enumerate(scenario["targets"]):
        end = scenario["targets"][segment_index + 1][0] if segment_index + 1 < len(scenario["targets"]) else scenario["duration"]
        first, last = round(start / 0.002), min(round(end / 0.002), len(actual))
        if first >= last:
            continue
        initial = np.asarray(scenario["initial"]) if first == 0 else actual[first - 1, :2]
        direction = np.sign(np.asarray(goal) - initial)
        overshoot = np.maximum(overshoot, np.maximum((actual[first:last, :2] - goal) * direction, 0).max(axis=0))
        within = np.all(np.abs(actual[first:last, :2] - goal) <= 0.02, axis=1) & np.all(np.abs(actual[first:last, 2:]) <= 0.05, axis=1)
        suffix = np.logical_and.accumulate(within[::-1])[::-1]
        candidates = np.flatnonzero(suffix & ((last - first - np.arange(last - first)) >= 150))
        settling = float((candidates[0] + 1) * 0.002) if len(candidates) else None
        segments.append({"target_start_s": start, "settling_time_s": settling})
    exceedance = np.maximum(np.maximum(joint_limits[:, 0] - actual[:, :2], actual[:, :2] - joint_limits[:, 1]), 0)
    predicted_errors = [np.asarray(diagnostic["predicted_next"]) - actual[(index + 1) * 50 - 1] for index, diagnostic in enumerate(episode["diagnostics"]) if "predicted_next" in diagnostic]
    timing = episode["timings"] * 1000
    return {
        "tracking_rmse_deg": np.rad2deg(np.sqrt(np.mean(difference ** 2, axis=0))).tolist(),
        "final_error_deg": np.rad2deg(np.abs(difference[-1])).tolist(),
        "settling_by_target": segments,
        "final_target_settling_s": segments[-1]["settling_time_s"],
        "maximum_overshoot_deg": np.rad2deg(overshoot).tolist(),
        "residual_velocity_rad_s": np.abs(actual[-1, 2:]).tolist(),
        "limit_violation_samples": int(np.count_nonzero(np.any(exceedance > 0.001, axis=1))),
        "maximum_limit_exceedance_deg": np.rad2deg(exceedance.max(axis=0)).tolist(),
        "active_limit_physics_steps": episode["active_limit_physics_steps"],
        "computation_ms": {"mean": float(timing.mean()), "p95": float(np.percentile(timing, 95)), "max": float(timing.max())},
        "deadline_misses": int(np.count_nonzero(timing > DT * 1000)),
        "fallback_steps": sum(item["fallback"] for item in episode["diagnostics"]),
        "goal_candidate_selected_steps": sum(item["selected"] == 0 for item in episode["diagnostics"]),
        "command_differs_from_goal_steps": int(np.count_nonzero(np.any(np.abs(episode["commands"] - episode["goals"]) > 1e-6, axis=1))),
        "executed_prediction_rmse": np.sqrt(np.mean(np.asarray(predicted_errors) ** 2, axis=0)).tolist() if predicted_errors else None,
    }


def plot_comparison(results, name, output):
    figure = Figure(figsize=(13, 14), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(5, 2)
    for controller, episode in results.items():
        times = np.arange(len(episode["states"])) * DT
        for joint in range(2):
            axes[0, joint].plot(times, np.rad2deg(episode["states"][:, joint]), label=controller)
            axes[1, joint].step(times[:-1], np.rad2deg(episode["commands"][:, joint]), where="post", label=controller)
            axes[2, joint].plot(times, episode["states"][:, joint + 2], label=controller)
            axes[3, joint].plot(np.arange(1, len(episode["fine_states"]) + 1) * 0.002, np.rad2deg(episode["fine_states"][:, joint] - episode["fine_goals"][:, joint]), label=controller)
        axes[4, 0].plot(times[:-1], episode["timings"] * 1000, label=controller)
    reference = results["direct"]
    for joint, label in enumerate(("shoulder", "elbow")):
        for row in (0, 1):
            axes[row, joint].step(np.arange(len(reference["goals"])) * DT, np.rad2deg(reference["goals"][:, joint]), where="post", color="black", linestyle="--", label="desired")
        for row, units in enumerate(("angle (deg)", "command (deg)", "velocity (rad/s)", "tracking error (deg)")):
            axes[row, joint].set(title=f"{label}: {units}", xlabel="Time (s)")
    axes[4, 0].axhline(100, color="black", linestyle=":", label="100 ms budget")
    axes[4, 0].set(title="Controller computation (ms)", xlabel="Time (s)")
    axes[4, 1].axis("off")
    for panel in axes.flat:
        if panel.axison:
            panel.legend()
    figure.suptitle(name)
    figure.savefig(output / f"{name}.png")


def audit_initial_plan(scenario, predict, config):
    """Compare the selected plan and direct goal in independent physics replays."""
    simulation = initialize_simulation(scenario["initial"])
    initial = state(simulation)
    previous = initial[:2]
    goal = np.asarray(scenario["targets"][0][1])
    joint_limits = simulation.model.jnt_range[simulation.joint_ids]
    planner = LearnedMPC(predict, simulation.action_limits, joint_limits, config)
    _, diagnostic = planner.act(initial, goal, previous)
    if diagnostic["fallback"]:
        return {"fallback": True}
    sequences = np.stack([planner.best, np.tile(goal, (config.horizon, 1))])
    predicted = rollout(predict, initial, sequences)
    actual = []
    for commands in sequences:
        replay = initialize_simulation(scenario["initial"])
        trajectory = []
        for command in commands:
            for _ in range(50):
                replay.step(command)
            trajectory.append(state(replay))
        actual.append(trajectory)
    predicted_costs = trajectory_cost(predicted, sequences, goal, previous, joint_limits, config)
    actual_costs = trajectory_cost(np.asarray(actual), sequences, goal, previous, joint_limits, config)
    return {
        "order": ["selected", "direct"],
        "predicted_costs": [float(value) if np.isfinite(value) else None for value in predicted_costs],
        "physics_costs": [float(value) if np.isfinite(value) else None for value in actual_costs],
        "ranking_reversal": bool(predicted_costs[0] < predicted_costs[1] and actual_costs[0] > actual_costs[1]),
        "selected_plan_state_rmse": np.sqrt(np.mean((predicted[0] - actual[0]) ** 2, axis=0)).tolist(),
    }


def benchmark(checkpoint, output, config):
    checkpoint, output = Path(checkpoint), Path(output)
    before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    predict = load_dynamics(checkpoint)
    predict(np.zeros((config.candidates, 4)), np.zeros((config.candidates, 2)))
    output.mkdir(parents=True, exist_ok=True)
    simulation = ArmSimulation()
    limits = simulation.model.jnt_range[simulation.joint_ids]
    report = {"config": asdict(config), "checkpoint_sha256": before, "scenarios": {}, "aggregate": {}}
    all_results = {"mpc": [], "direct": []}
    for name, scenario in scenarios().items():
        results = {controller: run_episode(scenario, controller, predict, config) for controller in ("direct", "mpc")}
        np.testing.assert_array_equal(results["direct"]["states"][0], results["mpc"]["states"][0])
        report["scenarios"][name] = {controller: metrics(episode, scenario, limits) for controller, episode in results.items()}
        report["scenarios"][name]["initial_plan_audit"] = audit_initial_plan(scenario, predict, config)
        for controller, episode in results.items():
            np.savez_compressed(output / f"{name}_{controller}.npz", **{key: value for key, value in episode.items() if isinstance(value, np.ndarray)})
            (output / f"{name}_{controller}_diagnostics.json").write_text(json.dumps(episode["diagnostics"], default=lambda value: value.tolist()), encoding="utf-8")
            all_results[controller].append(episode)
        plot_comparison(results, name, output)
        print(name, {controller: {key: value for key, value in report["scenarios"][name][controller].items() if key in ("tracking_rmse_deg", "final_error_deg", "final_target_settling_s")} for controller in results}, flush=True)
    for controller, episodes in all_results.items():
        summaries = [report["scenarios"][name][controller] for name in scenarios()]
        difference = np.concatenate([episode["fine_states"][:, :2] - episode["fine_goals"] for episode in episodes])
        times = np.concatenate([episode["timings"] for episode in episodes]) * 1000
        report["aggregate"][controller] = {
            "pooled_tracking_rmse_deg": np.rad2deg(np.sqrt(np.mean(difference ** 2, axis=0))).tolist(),
            "mean_final_error_deg": np.mean([summary["final_error_deg"] for summary in summaries], axis=0).tolist(),
            "settled_final_targets": sum(summary["final_target_settling_s"] is not None for summary in summaries),
            "mean_settling_s_among_settled": float(np.mean([summary["final_target_settling_s"] for summary in summaries if summary["final_target_settling_s"] is not None])) if any(summary["final_target_settling_s"] is not None for summary in summaries) else None,
            "max_overshoot_deg": np.max([summary["maximum_overshoot_deg"] for summary in summaries], axis=0).tolist(),
            "mean_residual_velocity_rad_s": np.mean([summary["residual_velocity_rad_s"] for summary in summaries], axis=0).tolist(),
            "limit_violation_samples": sum(summary["limit_violation_samples"] for summary in summaries),
            "active_limit_physics_steps": sum(summary["active_limit_physics_steps"] for summary in summaries),
            "computation_ms": {"mean": float(times.mean()), "p95": float(np.percentile(times, 95)), "max": float(times.max())},
            "deadline_misses": int(np.count_nonzero(times > 100)),
            "fallback_steps": sum(summary["fallback_steps"] for summary in summaries),
        }
    report["scenario_definitions"] = scenarios()
    report["versions"] = {"torch": str(torch.__version__), "mujoco": mujoco.__version__, "numpy": np.__version__}
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != before:
        raise RuntimeError("Checkpoint changed")
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/arm_dynamics_broad/model.pt"))
    parser.add_argument("--output", type=Path, default=Path("runs/mpc_initial"))
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--candidates", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weights", type=Path, help="JSON object of Config overrides")
    parser.add_argument("--viewer", choices=["mpc", "direct"])
    parser.add_argument("--scenario", choices=list(scenarios()), default="moving_target")
    args = parser.parse_args()
    overrides = json.loads(args.weights.read_text(encoding="utf-8")) if args.weights else {}
    config = Config(**({"horizon": args.horizon, "candidates": args.candidates, "seed": args.seed} | overrides))
    torch.set_num_threads(2)
    if args.viewer:
        try:
            run_episode(scenarios()[args.scenario], args.viewer, load_dynamics(args.checkpoint), config, viewer=True)
        except KeyboardInterrupt:
            pass
    else:
        benchmark(args.checkpoint, args.output, config)


if __name__ == "__main__":
    main()
