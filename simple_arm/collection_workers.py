"""Deterministic independent collection tasks using native Windows processes."""

import multiprocessing as mp
import os
import signal

import numpy as np


DEFAULT_WORKERS = min(8, os.cpu_count() or 1)
RNG_SCHEME = "seed_sequence_per_unit_v1"
_simulation = None


def unit_generator(seed, index, stream):
    """Separate task and split streams; scheduling never consumes randomness."""
    return np.random.default_rng(np.random.SeedSequence([seed, stream, index]))


def get_simulation():
    global _simulation
    if _simulation is None:
        from .contact_simulation import ContactSimulation
        _simulation = ContactSimulation()
    return _simulation


def initialize_worker():
    # The parent owns cancellation and terminates the entire pool on Ctrl+C.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    import torch
    torch.set_num_threads(1)


def effective_workers(workers, tasks):
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        raise ValueError("workers must be a positive integer")
    return min(workers, tasks)


def collect_tasks(function, tasks, workers=DEFAULT_WORKERS):
    """Yield completed tasks; parent assigns stable indices and writes files.

    The pool context terminates children on task failure or Ctrl+C. A task owns
    all of its rejection sampling and, for boundary data, all four related cases.
    """
    tasks = list(tasks)
    count = effective_workers(workers, len(tasks))
    if not tasks:
        return
    if workers == 1:
        yield from map(function, tasks)
    else:
        with mp.get_context("spawn").Pool(count, initializer=initialize_worker) as pool:
            yield from pool.imap_unordered(function, tasks, chunksize=1)
