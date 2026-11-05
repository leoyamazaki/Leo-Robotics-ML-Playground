"""Interactive target changes and decoration-only prediction geometry."""

from queue import Empty, SimpleQueue
import time

import mujoco
import numpy as np

from .cube_mpc import CEMController, DT, ModelAdapter, validate_command_rate
from .cube_mpc_experiment import execute_first, initialize, load_cases


def spawn_overlaps_robot(initial):
    simulation = initialize(initial)
    ground = simulation.model.geom("ground").id
    return any(contact.dist <= 0 and simulation.cube_geom in (contact.geom1, contact.geom2)
               and ground not in (contact.geom1, contact.geom2) for contact in simulation.data.contact)


def spawn_scenario(base, demonstrated_target, target=None, cube=None, spawn_seed=None):
    """Viewer-only placement; preserve the benchmark's saved states and targets."""
    if cube is not None and spawn_seed is not None:
        raise ValueError("Choose --cube or --spawn-seed")
    if spawn_seed is not None and (type(spawn_seed) is not int or spawn_seed < 0):
        raise ValueError("spawn-seed must be a nonnegative integer")
    initial = np.asarray(base, dtype=float).copy()
    explicit_target = None if target is None else np.asarray(target, dtype=float)
    if explicit_target is not None and (explicit_target.shape != (2,) or not np.isfinite(explicit_target).all()):
        raise ValueError("Target must contain two finite coordinates")
    goal_offset = np.asarray(demonstrated_target) - initial[4:6]
    if spawn_seed is not None:
        import torch
        from .cube_workspace import Workspace
        from .contact_varied import tip_position
        w = Workspace()
        rng = np.random.default_rng(spawn_seed)
        limits = w.limits.numpy()
        origin = w.origin.numpy()
        tip = tip_position(initial[:2], w.lengths) + origin
        # Leave one whole object/pusher footprint inside the outer radial reach;
        # use IK-derived tip distance too, including joint angular restrictions.
        radius_max = sum(w.lengths) - w.contact_radius
        separation_min = w.outer_contact_bound - sum(w.lengths) + .01
        for _ in range(2000):
            pose = rng.uniform(limits[:, 0] + .15, limits[:, 1] - .15)
            candidate = tip_position(pose, w.lengths) + origin
            if np.linalg.norm(candidate - origin) > radius_max or not separation_min <= np.linalg.norm(candidate - tip) <= .2:
                continue
            goal = candidate + goal_offset if explicit_target is None else explicit_target.copy()
            if explicit_target is None and w.tip_distance(torch.tensor(goal, dtype=torch.float32)).item() > 1e-6:
                continue
            initial[4:6] = candidate
            initial[9:] = 0
            if not spawn_overlaps_robot(initial):
                return initial, goal
        raise RuntimeError("Could not sample a close spawn; use --cube X Y")
    if cube is not None:
        cube = np.asarray(cube, dtype=float)
        if cube.shape != (2,) or not np.isfinite(cube).all():
            raise ValueError("Cube spawn must contain two finite coordinates")
        initial[4:6] = cube
        initial[9:] = 0
        if spawn_overlaps_robot(initial):
            raise ValueError("Cube spawn overlaps the robot; choose another --cube position")
    goal = ((np.asarray(demonstrated_target).copy() if cube is None else initial[4:6] + goal_offset)
            if explicit_target is None else explicit_target.copy())
    return initial, goal


def apply_key(key, target):
    target = np.asarray(target, dtype=float).copy()
    changes = {ord("6"): (0, -.02), ord("7"): (0, .02), ord("8"): (1, -.02), ord("9"): (1, .02)}
    if key in changes:
        axis, amount = changes[key]
        target[axis] += amount
    return target, key in changes, key in (13, 257)


def draw(scene, target, predicted):
    scene.ngeom = 0

    def geom(kind, size, position, color, rotation=None):
        if scene.ngeom >= scene.maxgeom:
            return
        item = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(item, kind, np.asarray(size), np.asarray(position),
                          np.eye(3).ravel() if rotation is None else rotation, np.asarray(color))
        item.category = mujoco.mjtCatBit.mjCAT_DECOR
        scene.ngeom += 1

    geom(mujoco.mjtGeom.mjGEOM_CYLINDER, [.02, .003, 0], [*target, .004], [0, 1, .1, .8])
    if predicted is None or not np.isfinite(predicted).all():
        return
    for point in predicted:
        geom(mujoco.mjtGeom.mjGEOM_SPHERE, [.004, 0, 0], point[4:7], [.9, .1, .8, .8])
    endpoint = predicted[-1]
    yaw = np.arctan2(endpoint[7], endpoint[8])
    rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]).ravel()
    geom(mujoco.mjtGeom.mjGEOM_BOX, [.035] * 3, endpoint[4:7], [.9, .1, .8, .3], rotation)


def view(checkpoint, dataset, episode, target, mode, config, seconds=None, *, cube=None, spawn_seed=None):
    if seconds is not None and (not np.isfinite(seconds) or seconds <= 0):
        raise ValueError("seconds must be positive and finite")
    base, demonstrated_target = load_cases(dataset)[episode]
    requested_target = target
    initial, original_target = spawn_scenario(base, demonstrated_target, requested_target, cube, spawn_seed)
    target = original_target.copy()
    simulation = initialize(initial)
    predict = ModelAdapter(checkpoint) if mode == "cem" else None
    planner = CEMController(predict, simulation.action_limits, simulation.model.jnt_range[simulation.joint_ids], config) if predict else None
    keys = SimpleQueue()
    previous, observed = initial[:2].copy(), simulation.extract()
    started = time.perf_counter()
    import mujoco.viewer
    print("6/7: target X -/+2cm; 8/9: target Y -/+2cm; Enter: reset; R: new close random cube spawn.", flush=True)
    print("Green: target; magenta: selected neural trajectory and unverified endpoint forecast.", flush=True)
    print(f"Joint-command rate limit: {config.command_rate_limit:g} rad/s (0 = unlimited)", flush=True)
    print(f"Cube XY: {initial[4:6].tolist()}; target: {target.tolist()}; spawn seed: {spawn_seed}", flush=True)
    with mujoco.viewer.launch_passive(simulation.model, simulation.data, key_callback=keys.put) as handle:
        with handle.lock():
            handle.cam.lookat[:] = [.2, -.1, .04]
            handle.cam.distance = 1.6
            handle.cam.elevation = -55
        while handle.is_running() and (seconds is None or time.perf_counter() - started < seconds):
            tick = time.perf_counter()
            while True:
                try:
                    key = keys.get_nowait()
                except Empty:
                    break
                target, changed, reset = apply_key(key, target)
                if key in (ord("R"), ord("r")):
                    spawn_seed = config.seed if spawn_seed is None else spawn_seed + 1
                    initial, original_target = spawn_scenario(base, demonstrated_target, requested_target, spawn_seed=spawn_seed)
                    reset = True
                    print(f"Spawn seed {spawn_seed}; cube XY {initial[4:6].tolist()}", flush=True)
                if reset:
                    from .contact_viewer import set_display_state
                    with handle.lock():
                        simulation.reset()
                        set_display_state(simulation, simulation.data, initial)
                        simulation.data.ctrl[simulation.actuator_ids] = initial[:2]
                    observed, previous = simulation.extract(), initial[:2].copy()
                    target = original_target.copy()
                if changed or reset:
                    if planner:
                        planner.reset()
                    print(f"Target {target.tolist()}", flush=True)
            planning_tick = time.perf_counter()
            plan = planner.plan(observed, target, previous) if planner else None
            elapsed = time.perf_counter() - planning_tick
            if planner and elapsed > DT:
                print(f"Planning overrun: {elapsed * 1000:.1f} ms (100 ms deadline); simulation interval stays 100 ms", flush=True)
            if plan is not None and plan.diagnostics["fallback"]:
                print("Fallback: all predictions infeasible; holding previous legal command", flush=True)
            command = plan.actions[0].copy() if plan is not None else initial[:2].copy()
            validate_command_rate(command, previous, config.command_rate_limit)
            previous = command
            with handle.lock():
                observed, _ = execute_first(simulation, previous)
                draw(handle.user_scn, target, plan.predicted if plan is not None else None)
            handle.sync()
            time.sleep(max(0, DT - (time.perf_counter() - tick)))
