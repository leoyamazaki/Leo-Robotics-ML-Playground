"""Frozen neural CEM cube controller. Run with ``python -m simple_arm.cube_mpc``."""

import argparse
from dataclasses import asdict, dataclass
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .contact_learning import BASE_DT, differentiable_next
from .contact_models import model_from_checkpoint
from .contact_simulation import link_lengths, model_hash


CHECKPOINT = Path("runs/arm_cube_capacity_refinement/medium_256/model.pt")
DATASET = Path("data/arm_cube_contact_boundary/independent.npz")
DT = 0.1
STATE_CONVENTION = "arm q2, arm v2, cube XYZ3, sin_yaw, cos_yaw, cube world v3, yaw_rate"
ACTION_CONVENTION = "absolute shoulder/elbow position targets in radians; held 100 ms"


@dataclass(frozen=True)
class Config:
    horizon: int = 15
    candidates: int = 256
    iterations: int = 5
    elite_fraction: float = .1
    knots: int = 5
    initial_std: float = .5
    retention: float = .2
    std_floor: float = .05
    command_margin: float = .15
    distance_weight: float = 1.
    terminal_weight: float = 5.
    change_weight: float = .01
    velocity_weight: float = .001
    seed: int = 42
    scoring: str = "original"
    reachability_weight: float = 10.
    cube_speed_weight: float = .1
    escape_velocity_weight: float = 1.
    terminal_reachability_weight: float = 20.
    terminal_cube_velocity_weight: float = .2
    free_cube_speed: float = .25
    command_rate_limit: float = .2  # rad/s per joint; zero explicitly disables limiting

    def __post_init__(self):
        for key in ("horizon", "candidates", "iterations", "knots", "seed"):
            value = getattr(self, key)
            if type(value) is not int or value < (0 if key == "seed" else 1):
                raise ValueError(f"Invalid {key}")
        if self.candidates < 2 or self.knots > self.horizon:
            raise ValueError("Require >=2 candidates and knots <= horizon")
        if self.scoring not in ("original", "workspace"):
            raise ValueError("scoring must be original or workspace")
        values = [v for k, v in asdict(self).items() if k not in ("horizon", "candidates", "iterations", "knots", "seed", "scoring")]
        if not np.isfinite(values).all() or min(values) < 0:
            raise ValueError("Settings must be finite and nonnegative")
        if not 0 < self.elite_fraction <= 1 or not 0 <= self.retention < 1:
            raise ValueError("Invalid elite fraction or retention")
        if self.std_floor <= 0 or self.initial_std < self.std_floor:
            raise ValueError("Require initial_std >= std_floor > 0")
        if self.free_cube_speed <= 0:
            raise ValueError("free_cube_speed must be positive")

    @property
    def elites(self):
        return math.ceil(self.candidates * self.elite_fraction)


class ModelAdapter:
    """CPU tensor inference using the training code's exact state transition."""

    def __init__(self, path=CHECKPOINT):
        torch.set_num_threads(2)
        self.checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        c, m = self.checkpoint, self.checkpoint["metadata"]
        if (c["stride"] != 5 or m.get("base_dt") != BASE_DT
                or m.get("physics_dt") != .002 or m.get("model_hash") != model_hash()
                or m.get("state") != STATE_CONVENTION or m.get("action") != ACTION_CONVENTION
                or m.get("lengths") != link_lengths()):
            raise ValueError("Checkpoint conventions, physics or timestep differ")
        for name, width in (("input_mean", 15), ("input_scale", 15), ("target_mean", 12), ("target_scale", 12)):
            value = c[name]
            if value.shape != (width,) or not torch.isfinite(value).all():
                raise ValueError(f"Invalid normalization: {name}")
            if name.endswith("scale") and not (value > 0).all():
                raise ValueError("Normalization scales must be positive")
        self.model = model_from_checkpoint(c).eval()
        self.model.load_state_dict(c["weights"])
        self.model.requires_grad_(False)

    def __call__(self, current, action):
        return differentiable_next(self.model, current, action, self.checkpoint)


def interpolate(values, steps):
    """Uniform linear knots, including both endpoints (also supports one knot)."""
    return F.interpolate(values.transpose(-1, -2), size=steps, mode="linear", align_corners=True).transpose(-1, -2)


def rollout(predict, observed, actions):
    current = observed.expand(len(actions), -1)
    predicted = []
    for action in actions.unbind(1):
        current = predict(current, action)
        predicted.append(current)
    return torch.stack(predicted, dim=1)


def limit_command_rate(actions, previous, rate_limit, dt=DT):
    """Project targets sequentially from the last executed command, in radians.

    Projection occurs before neural rollout and scoring. No post-plan clipping
    can make the executed command differ from the one the model evaluated.
    """
    if rate_limit == 0:
        return actions
    maximum_change = rate_limit * dt
    current = previous.expand(len(actions), 2)
    limited = []
    for requested in actions.unbind(1):
        current = current + (requested - current).clamp(-maximum_change, maximum_change)
        limited.append(current)
    return torch.stack(limited, dim=1)


def validate_command_rate(command, previous, rate_limit, dt=DT):
    """Execution guard: reject rather than silently alter a selected action."""
    if rate_limit and np.any(np.abs(np.asarray(command) - previous) > rate_limit * dt + 1e-6):
        raise RuntimeError("Selected command violates configured command-rate limit")


def score_components(predicted, actions, target, previous, config, workspace=None):
    distance = (predicted[..., 4:6] - target).square().sum(-1) / .1**2
    changes = torch.diff(torch.cat([previous.expand(len(actions), 1, 2), actions], 1), dim=1)
    components = {"target": config.distance_weight * distance.mean(1),
                  "terminal_target": config.terminal_weight * distance[:, -1],
                  "command_changes": config.change_weight * changes.square().mean((1, 2)),
                  "joint_velocity": config.velocity_weight * predicted[..., 2:4].square().mean((1, 2))}
    if config.scoring == "workspace":
        if workspace is None:
            raise ValueError("Workspace scoring requires robot geometry")
        components.update(workspace.cost_components(predicted, config, DT))
    return components


def trajectory_cost(predicted, actions, target, previous, joint_limits, config, workspace=None):
    if workspace is None and config.scoring == "workspace":
        from .cube_workspace import Workspace
        workspace = Workspace(joint_limits)
    terms = score_components(predicted, actions, target, previous, config, workspace)
    # Preserve the original summation order for exact baseline reproducibility.
    cost = terms["target"] + terms["terminal_target"] + terms["command_changes"] + terms["joint_velocity"]
    for name in ("workspace", "excess_cube_speed", "escape_velocity", "terminal_workspace", "terminal_cube_velocity"):
        if name in terms:
            cost = cost + terms[name]
    finite = torch.isfinite(predicted).all(dim=(1, 2)) & torch.isfinite(cost)
    legal = ((predicted[..., :2] >= joint_limits[:, 0]) & (predicted[..., :2] <= joint_limits[:, 1])).all(dim=(1, 2))
    return cost.masked_fill(~(finite & legal), torch.inf), ~finite, ~legal


def fit_distribution(mean, std, elites, config):
    retain = config.retention
    return (retain * mean + (1 - retain) * elites.mean(0),
            (retain * std + (1 - retain) * elites.std(0, correction=0)).clamp_min(config.std_floor))


@dataclass
class Plan:
    actions: np.ndarray
    predicted: np.ndarray
    cost: float | None
    diagnostics: dict


class CEMController:
    """No simulator, collision queries, contact labels or demonstration actions."""

    def __init__(self, predict, action_limits, joint_limits, config=Config()):
        self.predict, self.config = predict, config
        self.joint_limits = torch.as_tensor(joint_limits, dtype=torch.float32).clone()
        limits = torch.as_tensor(action_limits, dtype=torch.float32)
        self.lower = torch.maximum(limits[:, 0], self.joint_limits[:, 0] + config.command_margin)
        self.upper = torch.minimum(limits[:, 1], self.joint_limits[:, 1] - config.command_margin)
        if not torch.isfinite(torch.stack([self.lower, self.upper])).all() or (self.lower >= self.upper).any():
            raise ValueError("No legal command interval")
        from .cube_workspace import Workspace
        self.workspace = Workspace(self.joint_limits) if config.scoring == "workspace" else None
        self.reset()

    def reset(self):
        self.generator = torch.Generator(device="cpu").manual_seed(self.config.seed)
        self.best = None

    def warm_mean(self, previous):
        if self.best is None:
            return previous.expand(self.config.knots, 2).clone()
        shifted = torch.cat([self.best[1:], self.best[-1:]], dim=0)
        return interpolate(shifted[None], self.config.knots)[0]

    @torch.inference_mode()
    def plan(self, observed_state, target_xy, previous_command):
        c = self.config
        observed = torch.as_tensor(observed_state, dtype=torch.float32)
        target = torch.as_tensor(target_xy, dtype=torch.float32)
        previous = torch.as_tensor(previous_command, dtype=torch.float32)
        if observed.shape != (13,) or target.shape != (2,) or previous.shape != (2,) or not all(torch.isfinite(x).all() for x in (observed, target, previous)):
            raise ValueError("Require finite state13, target2 and previous2")
        # Permit only a legal hold command, including in the all-infeasible case.
        if ((previous < self.lower) | (previous > self.upper)).any():
            raise ValueError("Previous command outside safe command bounds")
        mean = self.warm_mean(previous)
        std = torch.full_like(mean, c.initial_std)
        initial_mean = mean.clone()
        winner, winner_prediction, winner_cost = None, None, float("inf")
        history = []
        for iteration in range(c.iterations):
            knots = mean + std * torch.randn(c.candidates, c.knots, 2, generator=self.generator)
            knots = knots.clamp(self.lower, self.upper)
            knots[0], knots[1] = mean, previous
            actions = interpolate(knots, c.horizon)
            actions = limit_command_rate(actions, previous, c.command_rate_limit)
            if c.command_rate_limit:
                actions = actions.clamp(self.lower, self.upper)
            predicted = rollout(self.predict, observed, actions)
            costs, nonfinite, limits = trajectory_cost(predicted, actions, target, previous, self.joint_limits, c, self.workspace)
            feasible = torch.isfinite(costs)
            count = int(feasible.sum())
            if count:
                selected = int(costs.argmin())
                if float(costs[selected]) < winner_cost:
                    winner_cost = float(costs[selected])
                    winner, winner_prediction = actions[selected].clone(), predicted[selected].clone()
                elite_indices = torch.topk(costs, min(c.elites, count), largest=False).indices
                mean, std = fit_distribution(mean, std, knots[elite_indices], c)
            history.append({"iteration": iteration, "feasible": count, "rejected": c.candidates - count,
                            "nonfinite": int(nonfinite.sum()), "joint_limit": int(limits.sum()),
                            "elite_count": min(c.elites, count),
                            "best_cost": winner_cost if winner is not None else None})
        fallback = winner is None
        if fallback:
            winner = previous.expand(c.horizon, 2).clone()
            # NaNs explicitly mean no feasible prediction, not a trusted hold forecast.
            winner_prediction = torch.full((c.horizon, 13), torch.nan)
            self.best = None
        else:
            self.best = winner.clone()
        diagnostics = {"fallback": fallback, "rejected": sum(h["rejected"] for h in history),
                       "iterations": history, "initial_mean": initial_mean.tolist(),
                       "final_mean": mean.tolist(), "final_std": std.tolist(),
                       "initial_std": c.initial_std}
        diagnostics["scoring"] = c.scoring
        diagnostics["command_rate_limit_rad_s"] = c.command_rate_limit
        diagnostics["selected_max_command_rate_rad_s"] = float(
            torch.diff(torch.cat([previous[None], winner], dim=0), dim=0).abs().max() / DT)
        diagnostics["score_components"] = ({} if fallback else {name: float(value[0]) for name, value in
            score_components(winner_prediction[None], winner[None], target, previous, c, self.workspace).items()})
        return Plan(winner.numpy().copy(), winner_prediction.numpy().copy(),
                    None if fallback else winner_cost, diagnostics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["evaluate", "compare", "compare-rate", "view"])
    parser.add_argument("--checkpoint", type=Path, help="Model path; compare-rate defaults to stationary_20, other commands to refined Medium")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output", type=Path, default=Path("runs/cube_cem_initial"))
    parser.add_argument("--episode", type=int, choices=[23, 99, 139, 159], default=23)
    parser.add_argument("--target", type=float, nargs=2)
    spawn = parser.add_mutually_exclusive_group()
    spawn.add_argument("--cube", type=float, nargs=2, metavar=("X", "Y"), help="Viewer cube spawn XY in meters")
    spawn.add_argument("--spawn-seed", type=int, help="Viewer random close/reachable cube spawn; separate from CEM --seed")
    parser.add_argument("--mode", choices=["cem", "no-action"], default="cem")
    parser.add_argument("--seconds", type=float, help="Viewer wall-clock duration for smoke tests")
    for name, value in asdict(Config()).items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(value), default=value,
                            help="Maximum joint-target slew in rad/s per joint (default: 0.2); 0 disables" if name == "command_rate_limit" else None,
                            **({"choices": ["original", "workspace"]} if name == "scoring" else {}))
    args = parser.parse_args()
    if args.command != "view" and (args.cube is not None or args.spawn_seed is not None):
        parser.error("--cube and --spawn-seed are viewer options")
    config = Config(**{name: getattr(args, name) for name in asdict(Config())})
    checkpoint = args.checkpoint or (Path("runs/arm_cube_stationary_experiment/stationary_20/model.pt")
                                     if args.command == "compare-rate" else CHECKPOINT)
    if args.command == "compare-rate":
        from .cube_rate_experiment import compare
        output = Path("runs/cube_cem_command_rate") if args.output == Path("runs/cube_cem_initial") else args.output
        compare(checkpoint, args.dataset, output, config)
    elif args.command == "compare":
        from .cube_mpc_comparison import compare
        output = Path("runs/cube_cem_workspace") if args.output == Path("runs/cube_cem_initial") else args.output
        compare(checkpoint, args.dataset, output, config)
    elif args.command == "evaluate":
        from .cube_mpc_experiment import evaluate
        evaluate(checkpoint, args.dataset, args.output, config)
    else:
        from .cube_mpc_viewer import view
        view(checkpoint, args.dataset, args.episode, args.target, args.mode, config, args.seconds,
             cube=args.cube, spawn_seed=args.spawn_seed)


if __name__ == "__main__":
    main()
