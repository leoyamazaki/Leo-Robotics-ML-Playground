"""Stationary-start contact data with smooth, rate-limited position commands."""

import time
from pathlib import Path

import numpy as np

from .collection_workers import collect_tasks, get_simulation, unit_generator
from .contact_boundary import boundary_offset, group_split, load_boundary, save_dataset, simulate
from .contact_simulation import link_lengths, model_hash
from .contact_varied import tip_position


KINDS = ["overhead", "near_miss", "gentle_push", "push_release"]


def commands(generator, initial, limits, rate=.2):
    """Slow ramps, pauses, reversals and settling; no initial command jump."""
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("rate must be positive and finite")
    low, high = limits[:, 0] + .15, limits[:, 1] - .15
    target = np.asarray(initial).copy()
    direction = generator.choice([-1., 1.], 2)
    speeds = generator.uniform(.65 * rate, rate, 2)
    velocity = np.zeros(2)
    stop = int(generator.integers(28, 33))
    reverse = generator.random() < .25
    result = []
    for step in range(40):
        wanted = speeds * direction if step < stop else np.zeros(2)
        if reverse and step >= 20 and step < stop:
            wanted = -wanted
        velocity += np.clip(wanted - velocity, -.1 * rate, .1 * rate)
        requested = np.clip(target + .1 * velocity, low, high)
        target += np.clip(requested - target, -.1 * rate, .1 * rate)
        result.append(target.copy())
    return np.asarray(result)


def _group(task):
    seed, group, rate = task
    rng = unit_generator(seed, group, 42)
    sim, lengths = get_simulation(), link_lengths()
    limits = sim.action_limits
    rejected = 0
    for attempt in range(60):
        arm = rng.uniform(limits[:, 0] + .25, limits[:, 1] - .25)
        control = commands(rng, arm, limits, rate)
        yaw = rng.uniform(-np.pi, np.pi)
        try:
            free = simulate(sim, arm, [.9, .9], yaw, np.zeros(2), control, diagnostics=False)
        except ValueError:
            continue
        qtrace = free[-1]
        path = tip_position(qtrace, lengths)
        index = int(rng.integers(650, 1200))
        tangent = path[index + 10] - path[index - 10]
        if np.linalg.norm(tangent) < .0005:
            continue
        normal = rng.choice([-1., 1.]) * np.array([-tangent[1], tangent[0]]) / np.linalg.norm(tangent)
        point = path[index]
        elbow = lengths[0] * np.array([np.cos(qtrace[index, 0]), np.sin(qtrace[index, 0])])
        try:
            edge = boundary_offset(path, point, normal, yaw, 0.)
            near = boundary_offset(path, point, normal, yaw, .002)
        except ValueError:
            continue
        candidates = [(0, f * elbow) for f in (.6, .8, .4)]
        candidates += [(0, elbow + .6 * (point - elbow)), (1, point + near * normal)]
        candidates += [(2, point + factor * edge * normal) for factor in (0., .15, .3)]
        candidates += [(3, point + factor * edge * normal) for factor in (.55, .7, .4)]
        found = {}
        for kind, cube in candidates:
            if kind in found or np.linalg.norm(cube) < .12:
                continue
            try:
                result = simulate(sim, arm, cube, yaw, np.zeros(2), control, diagnostics=False)
            except ValueError:
                rejected += 1
                continue
            state, contacts, starting, diagnostic, trace = result
            motion = np.linalg.norm(state[:, 4:6] - state[:1, 4:6], axis=-1).max()
            active = int(contacts.sum())
            acceptable = ((kind <= 1 and active == 0 and motion <= .002)
                          or (kind == 2 and active >= 50 and motion > .02)
                          or (kind == 3 and active > 0 and motion > .005))
            if acceptable:
                found[kind] = (state, contacts, starting, float(np.abs(state[:, 2:4]).max()))
            else:
                rejected += 1
        if len(found) == 4:
            return group, found, np.repeat(control, 5, axis=0), {"attempts": attempt + 1, "rejected": rejected,
                "initial_command": arm.tolist(), "command_peak_rate_rad_s": float(np.abs(np.diff(np.vstack([arm, control]), axis=0)).max() / .1)}
    raise RuntimeError(f"Slow group {group} failed after 60 approaches; rejected={rejected}")


def collect(path, groups=80, seed=394, rate=.2, workers=4):
    path = Path(path)
    if path.exists() or path.with_suffix(".json").exists():
        raise FileExistsError(f"Preserving existing dataset: {path}")
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("rate must be positive and finite")
    split = group_split(groups, unit_generator(seed, 0, 43))
    count = groups * 4
    states, actions = np.empty((count, 201, 13)), np.empty((count, 200, 2))
    contacts, starting = np.empty((count, 200, 10), dtype=bool), np.empty((count, 200), dtype=bool)
    specs = [None] * groups
    peak_speed = 0.
    tick = time.perf_counter()
    for completed, (group, found, control, spec) in enumerate(collect_tasks(_group, [(seed, g, rate) for g in range(groups)], workers), 1):
        specs[group] = spec
        for kind in range(4):
            index = 4 * group + kind
            states[index], contacts[index], starting[index], peak = found[kind]
            actions[index] = control
            peak_speed = max(peak_speed, peak)
        if completed % 10 == 0:
            print(f"Slow collection seed={seed}: {completed}/{groups} groups", flush=True)
    metadata = {"seed": seed, "episodes": count, "base_dt": .02, "physics_dt": .002,
                "model_hash": model_hash(), "lengths": link_lengths(), "kinds": KINDS,
                "collector": "slow_controlled_v1", "command_rate_limit_rad_s": rate,
                "reference_acceleration_rad_s2": rate, "stationary_starts": True,
                "state": "arm q2, arm v2, cube XYZ3, sin_yaw, cos_yaw, cube world v3, yaw_rate",
                "action": "absolute shoulder/elbow position targets in radians; held 100 ms",
                "selection": "paired stationary overhead/near-miss and two gentle-contact cases; no joint-limit excursions"}
    dataset = dict(states=states, actions=actions, contacts=contacts, starting_contact=starting,
                   split=split, kinds=np.tile(np.arange(4), groups), group_ids=np.repeat(np.arange(groups), 4), metadata=metadata)
    summary = {"groups": groups, "episodes": count, "split_episodes": np.bincount(split).tolist(),
               "rate_limit_rad_s": rate, "peak_command_rate_rad_s": max(s["command_peak_rate_rad_s"] for s in specs),
               "maximum_actual_joint_speed_rad_s": peak_speed,
               "maximum_actual_cube_speed_m_s": float(np.linalg.norm(states[..., 9:11], axis=-1).max()),
               "stationary_episode_fraction": float((np.linalg.norm(states[..., 4:6] - states[:, :1, 4:6], axis=-1).max(1) <= .002).mean()),
               "collection_seconds": time.perf_counter() - tick, "sampling": specs}
    save_dataset(path, dataset, summary)
    return load_boundary(path), summary
