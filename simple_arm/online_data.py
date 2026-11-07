"""Unfiltered random MuJoCo episodes, ordered asynchronous generation and replay."""

from collections import deque
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path

import mujoco
import numpy as np

from .collection_workers import get_simulation, initialize_worker, unit_generator
from .contact_simulation import link_lengths, model_hash
from .cube_mpc import ACTION_CONVENTION, STATE_CONVENTION


FAMILIES = ["idle", "shoulder_only", "elbow_only", "independent_goals", "coordinated", "reversals"]


def random_commands(rng, initial, limits, family, rate=.2):
    """Persistent random goals/rates; independently redrawn targets would jitter."""
    target = np.asarray(initial).copy()
    low, high = limits[:, 0] + .15, limits[:, 1] - .15
    goal, velocity, hold = target.copy(), np.zeros(2), 0
    result = []
    for step in range(40):
        if family == 0:
            goal = target.copy()
        elif family <= 3:
            if hold == 0:
                goal = rng.uniform(low, high)
                if family == 1:
                    goal[1] = initial[1]
                elif family == 2:
                    goal[0] = initial[0]
                hold = int(rng.integers(8, 25))
        else:
            if hold == 0:
                if family == 5 and step:
                    velocity *= -1
                else:
                    velocity = rng.uniform(.02, rate, 2) * rng.choice([-1., 1.], 2)
                    if family == 4:
                        velocity[1] = velocity[0] * rng.choice([-1., 1.])
                hold = int(rng.integers(8, 21))
            goal = np.clip(target + .1 * velocity, low, high)
        target += np.clip(goal - target, -.1 * rate, .1 * rate)
        result.append(target.copy())
        hold = max(0, hold - 1)
    return np.asarray(result)


def random_episode(task):
    """Retry invalid physical starts only. Never select by contact or displacement."""
    seed, index = task
    rng = unit_generator(seed, index, 71)
    sim = get_simulation()
    family = int(rng.integers(len(FAMILIES)))
    invalid = 0
    for _ in range(200):
        arm = rng.uniform(sim.action_limits[:, 0] + .15, sim.action_limits[:, 1] - .15)
        cube = rng.uniform(-.55, .55, 2)
        yaw = rng.uniform(-np.pi, np.pi)
        if np.linalg.norm(cube) < .12:
            invalid += 1
            continue
        moving = rng.random() < .5
        speed, angle = (rng.uniform(.02, .15) if moving else 0.), rng.uniform(-np.pi, np.pi)
        cube_velocity = [speed * np.cos(angle), speed * np.sin(angle), rng.uniform(-1, 1) if moving else 0.]
        try:
            initial = sim.initialize(arm, cube, yaw, np.zeros(2), cube_velocity)
            break
        except ValueError:
            invalid += 1
    else:
        raise RuntimeError(f"Random seed={seed} episode={index}: 200 invalid starts")
    commands = random_commands(rng, arm, sim.action_limits, family)
    states, actions = np.empty((201, 13)), np.repeat(commands, 5, axis=0)
    contacts, starting = np.empty((200, 10), dtype=bool), np.empty(200, dtype=bool)
    states[0] = initial
    peak, limit_excursion = 0., False
    for step in range(200):
        sim.data.ctrl[sim.actuator_ids] = actions[step]
        starting[step] = sim.contact()
        for substep in range(10):
            mujoco.mj_step(sim.model, sim.data)
            mujoco.mj_forward(sim.model, sim.data)
            contacts[step, substep] = sim.contact()
            peak = max(peak, float(np.abs(sim.data.qvel[sim.qvel_indices]).max()))
            q = sim.data.qpos[sim.qpos_indices]
            limit_excursion |= bool(np.any(q < sim.action_limits[:, 0] - 1e-4) or np.any(q > sim.action_limits[:, 1] + 1e-4))
        states[step + 1] = sim.extract()
    if not np.isfinite(states).all():
        raise RuntimeError(f"Nonfinite physics: seed={seed}, episode={index}; no outcome-based retry")
    spec = {"episode_id": index, "invalid_starts": invalid, "moving_cube_start": moving,
            "peak_joint_speed_2ms_rad_s": peak, "joint_limit_excursion": limit_excursion}
    return states, actions, contacts, starting, family, spec


class EpisodeStream:
    """Bounded prefetch; consume in index order so worker scheduling changes no data."""
    def __init__(self, seed, start, stop, workers=4, prefetch=1000):
        self.seed, self.next_index, self.stop = seed, start, stop
        self.executor = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"), initializer=initialize_worker)
        self.pending = deque()
        self.submitted = start
        for _ in range(min(prefetch, stop - start)):
            self._submit()

    def _submit(self):
        if self.submitted < self.stop:
            self.pending.append(self.executor.submit(random_episode, (self.seed, self.submitted)))
            self.submitted += 1

    def take(self, count):
        if count < 1 or self.next_index + count > self.stop:
            raise ValueError("Invalid stream request")
        rows = []
        for _ in range(count):
            rows.append(self.pending.popleft().result())
            self.next_index += 1
            self._submit()
        return rows

    def close(self):
        for future in self.pending:
            future.cancel()
        self.executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def metadata(seed, count):
    return {"seed": seed, "episodes": count, "base_dt": .02, "physics_dt": .002,
            "model_hash": model_hash(), "lengths": link_lengths(), "kinds": FAMILIES,
            "state": STATE_CONVENTION, "action": ACTION_CONVENTION, "collector": "uniform_online_v1",
            "command_rate_limit_rad_s": .2, "selection": "invalid physical starts only; no outcome selection",
            "cube_xy_bounds_m": [-.55, .55], "base_exclusion_radius_m": .12,
            "mujoco_version": mujoco.__version__, "numpy_version": np.__version__}


def dataset_from_rows(rows, seed, split=0):
    return {**{key: np.stack([row[i] for row in rows]) for i, key in enumerate(("states", "actions", "contacts", "starting_contact"))},
            "kinds": np.array([row[4] for row in rows]), "split": np.full(len(rows), split),
            "episode_ids": np.array([row[5]["episode_id"] for row in rows]),
            "metadata": dict(metadata(seed, len(rows)), sampling=[row[5] for row in rows])}


def save_archive(path, dataset):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **{k: v for k, v in dataset.items() if k != "metadata"}, metadata=json.dumps(dataset["metadata"]))
    temporary.replace(path)


def load_archive(path):
    with np.load(path, allow_pickle=False) as archive:
        result = {key: archive[key].copy() for key in archive.files if key != "metadata"}
        result["metadata"] = json.loads(str(archive["metadata"]))
    if result["metadata"]["model_hash"] != model_hash() or not np.isfinite(result["states"]).all():
        raise ValueError(f"Invalid random-data archive: {path}")
    return result


class ReplayBuffer:
    """Keep complete episodes in insertion order; random sampling happens in trainer."""
    def __init__(self, capacity=5000):
        self.capacity = capacity
        self.data = None

    def append(self, dataset):
        keys = ("states", "actions", "contacts", "starting_contact", "kinds", "episode_ids")
        self.data = {key: (dataset[key] if self.data is None else np.concatenate([self.data[key], dataset[key]]))[-self.capacity:]
                     for key in keys}
        self.data["split"] = np.zeros(len(self.data["states"]), dtype=int)
        self.data["metadata"] = dataset["metadata"]

    def __len__(self):
        return 0 if self.data is None else len(self.data["states"])
