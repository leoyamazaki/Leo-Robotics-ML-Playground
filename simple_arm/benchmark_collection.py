"""Quick serial versus process-pool timing of real collection physics checks."""

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import statistics
import time

import numpy as np


_simulation = None
_simulate = None


def initialize_worker(ready, warmup):
    global _simulation, _simulate
    import torch
    from .contact_boundary import simulate
    from .contact_simulation import ContactSimulation

    # Collection uses no neural network; avoid unrelated tensor thread pools.
    torch.set_num_threads(1)
    _simulation, _simulate = ContactSimulation(), simulate
    run_episode(warmup)
    if ready is not None:
        ready.put(os.getpid())


def run_episode(job):
    arm, cube, yaw, velocity, commands = job
    result = _simulate(_simulation, arm, cube, yaw, velocity, commands)
    digest = hashlib.sha256()
    for array in result:
        digest.update(array.tobytes())
    return digest.hexdigest()


def load_jobs(path, episodes):
    with np.load(path, allow_pickle=False) as archive:
        if episodes > len(archive["states"]):
            raise ValueError("Requested more episodes than available")
        initial = archive["states"][:episodes, 0].copy()
        commands = archive["actions"][:episodes, ::5].copy()
    return [(state[:2].copy(), state[4:6].copy(), float(np.arctan2(state[7], state[8])),
             state[2:4].copy(), action) for state, action in zip(initial, commands)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/arm_cube_contact_boundary/focused.npz"))
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--workers", type=int, nargs="+", default=[2, 4])
    args = parser.parse_args()
    if args.episodes < 4 or args.episodes % 4 or args.repeats < 1 or any(n < 2 for n in args.workers):
        parser.error("Use a positive multiple of four episodes, positive repeats, and worker counts >=2")
    jobs = load_jobs(args.dataset, args.episodes)
    rows = []
    started = time.perf_counter()
    initialize_worker(None, jobs[0])
    startup = time.perf_counter() - started
    reference = None
    timings = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        result = list(map(run_episode, jobs))
        timings.append(time.perf_counter() - started)
        if reference is None:
            reference = result
        elif result != reference:
            raise RuntimeError("Serial collection was not deterministic")
    rows.append({"workers": 1, "startup_and_warmup_s": startup, "trials_s": timings})
    print(json.dumps(rows[-1]), flush=True)
    context = mp.get_context("spawn")
    for workers in args.workers:
        ready = context.Queue()
        started = time.perf_counter()
        with context.Pool(workers, initializer=initialize_worker, initargs=(ready, jobs[0])) as pool:
            initialized = {ready.get(timeout=120) for _ in range(workers)}
            if len(initialized) != workers:
                raise RuntimeError("Not every worker completed initialization")
            startup = time.perf_counter() - started
            timings = []
            for _ in range(args.repeats):
                started = time.perf_counter()
                result = pool.map(run_episode, jobs, chunksize=1)
                timings.append(time.perf_counter() - started)
                if result != reference:
                    raise RuntimeError("Parallel results differ from serial physics")
        ready.close()
        ready.join_thread()
        rows.append({"workers": workers, "startup_and_warmup_s": startup, "trials_s": timings})
        print(json.dumps(rows[-1]), flush=True)
    baseline = statistics.median(rows[0]["trials_s"])
    for row in rows:
        row["median_s"] = statistics.median(row["trials_s"])
        row["episodes_per_s"] = args.episodes / row["median_s"]
        row["steady_speedup"] = baseline / row["median_s"]
        row["startup_plus_first_batch_s"] = row["startup_and_warmup_s"] + row["trials_s"][0]
    print("SUMMARY " + json.dumps({"episodes_per_trial": args.episodes, "repeats": args.repeats,
        "scope": "Accepted episodes: exact 2 ms physics/contact/distance checks; excludes placement search, rejection and file writing",
        "all_results_identical": True, "results": rows}), flush=True)


if __name__ == "__main__":
    main()
