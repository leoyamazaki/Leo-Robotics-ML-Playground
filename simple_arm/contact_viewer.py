"""Replay recorded MuJoCo motion with an autoregressive prediction ghost."""

import argparse
import json
from pathlib import Path
from queue import Empty, SimpleQueue
import time

import mujoco
import numpy as np

from .contact_simulation import ContactSimulation


DEFAULT_RUN = Path(__file__).resolve().parents[1] / "runs/arm_cube_varied_unseen_recursive"
DEFAULT_CHECKPOINT = DEFAULT_RUN.parent / "arm_cube_dynamics_varied_recursive/model.pt"


def load_replay(directory):
    directory = Path(directory)
    with np.load(directory / "test_rollouts.npz", allow_pickle=False) as archive:
        actual, predicted = archive["actual"].copy(), archive["learned"].copy()
    dt = float(json.loads((directory / "evaluation.json").read_text())["dt"])
    if (actual.shape != predicted.shape or actual.ndim != 3 or actual.shape[-1] != 13
            or actual.shape[0] < 1 or actual.shape[1] < 2 or not np.isfinite(dt) or dt <= 0
            or not np.isfinite(actual).all() or not np.isfinite(predicted).all()):
        raise ValueError("Invalid saved contact rollout")
    if not np.allclose(actual[:, 0], predicted[:, 0]):
        raise ValueError("Replay must start prediction and physics from the same state")
    return actual, predicted, dt


def eligible_episodes(actual, condition="any", threshold=0.002):
    excursion = np.linalg.norm(actual[..., 4:6] - actual[:, :1, 4:6], axis=-1).max(axis=1)
    indices = np.flatnonzero(excursion <= threshold) if condition == "still" else np.arange(len(actual))
    if not len(indices):
        raise ValueError(f"No episodes match {condition!r} at threshold {threshold:g} m")
    return indices


def choose_episode(actual, predicted, choice, threshold=0.002):
    if choice == "worst-still":
        candidates = eligible_episodes(actual, "still", threshold)
        errors = np.mean(np.sum((predicted - actual)[..., 4:6] ** 2, axis=-1), axis=1)
        return int(candidates[np.argmax(errors[candidates])])
    if choice in ("best", "worst"):
        moving = np.flatnonzero(np.linalg.norm(actual[:, -1, 4:6] - actual[:, 0, 4:6], axis=-1) > 0.02)
        candidates = moving if len(moving) else np.arange(len(actual))
        errors = np.mean(np.sum((predicted - actual)[..., 4:6] ** 2, axis=-1), axis=1)
        pick = np.argmin if choice == "best" else np.argmax
        return int(candidates[pick(errors[candidates])])
    index = int(choice)
    if not 0 <= index < len(actual):
        raise ValueError(f"Episode must be between 0 and {len(actual) - 1}")
    return index


def generate_random_episode(seed, predict, condition="any", threshold=0.002, max_attempts=100):
    """Generate physics first, then an independent recursive prediction from t=0.

    Still filtering uses simulator outcomes only for selecting examples to watch.
    No selection by prediction quality or corrections from later truth states.
    """
    from .contact_varied import FAMILIES, command_sequence, tip_position
    from .contact_simulation import link_lengths
    from .contact_learning import rollout

    generator = np.random.default_rng(seed)
    simulation = ContactSimulation()
    limits, lengths = simulation.action_limits, link_lengths()
    for attempt in range(max_attempts):
        arm = generator.uniform(limits[:, 0] + 0.15, limits[:, 1] - 0.15)
        family = int(generator.integers(4))
        actions = command_sequence(generator, arm, limits, family)
        # No contact-enrichment or displacement acceptance gate in exploratory mode.
        # Still mode samples reachable placements independently of the arm path.
        if condition == "any" and generator.random() < 0.7:
            cube = tip_position(actions[generator.integers(1, 8)], lengths) + generator.uniform(-0.015, 0.015, 2)
        else:
            cube = tip_position(generator.uniform(limits[:, 0] + 0.15, limits[:, 1] - 0.15), lengths)
        if np.linalg.norm(cube) < 0.12:
            continue
        cube_velocity = np.zeros(3)
        if condition == "any" and generator.random() < 0.5:
            angle, speed = generator.uniform(-np.pi, np.pi), generator.uniform(0.02, 0.15)
            cube_velocity = [speed * np.cos(angle), speed * np.sin(angle), generator.uniform(-1, 1)]
        try:
            arm_velocity = generator.uniform(-0.5, 0.5, 2) if generator.random() < 0.5 else np.zeros(2)
            arm_velocity = np.where((arm < limits[:, 0] + 0.3) | (arm > limits[:, 1] - 0.3), 0, arm_velocity)
            initial = simulation.initialize(arm, cube, generator.uniform(-np.pi, np.pi),
                                            arm_velocity, cube_velocity)
        except ValueError:
            continue
        actual = [initial]
        contact_steps = 0
        excursion = 0.0
        invalid = False
        for action in actions:
            simulation.data.ctrl[simulation.actuator_ids] = action
            for _ in range(50):
                mujoco.mj_step(simulation.model, simulation.data)
                mujoco.mj_forward(simulation.model, simulation.data)
                contact_steps += int(simulation.contact())
                q = simulation.data.qpos[simulation.qpos_indices]
                invalid |= bool(np.any(q < limits[:, 0] - 1e-4) or np.any(q > limits[:, 1] + 1e-4))
                excursion = max(excursion, float(np.linalg.norm(simulation.data.xpos[simulation.cube_id, :2] - initial[4:6])))
            actual.append(simulation.extract())
        actual = np.asarray(actual)
        if invalid or not np.isfinite(actual).all():
            continue
        if condition == "still" and excursion > threshold:
            continue
        predicted = rollout(predict, actual[None, 0], actions[None])[0]
        return actual, predicted, {"seed": seed, "family": FAMILIES[family], "attempts": attempt + 1,
                                  "actual_max_motion_mm": float(excursion * 1000),
                                  "contact_substeps": contact_steps}
    raise RuntimeError(f"No valid {condition} episode after {max_attempts} attempts for seed {seed}")


def sample_state(trajectory, seconds, dt):
    """Interpolate only for display, using the shortest yaw arc."""
    frame = np.clip(seconds / dt, 0, len(trajectory) - 1)
    index = min(int(frame), len(trajectory) - 2)
    fraction = frame - index
    left, right = trajectory[index:index + 2]
    result = (1 - fraction) * left + fraction * right
    yaw_left, yaw_right = np.arctan2(left[7], left[8]), np.arctan2(right[7], right[8])
    yaw_delta = np.arctan2(np.sin(yaw_right - yaw_left), np.cos(yaw_right - yaw_left))
    yaw = yaw_left + fraction * yaw_delta
    result[7:9] = [np.sin(yaw), np.cos(yaw)]
    return result


def set_display_state(simulation, data, state):
    """Map world cube coordinates to its slide joints; forward kinematics only."""
    data.qpos[simulation.qpos_indices] = state[:2]
    data.qvel[simulation.qvel_indices] = state[2:4]
    cube_origin = simulation.model.body_pos[simulation.cube_id]
    data.qpos[simulation.cube_q] = [*(state[4:7] - cube_origin), np.arctan2(state[7], state[8])]
    data.qvel[simulation.cube_v] = [*state[9:12], state[12]]
    mujoco.mj_forward(simulation.model, data)


def draw_prediction(scene, simulation, predicted_data):
    """Visual geoms only: the ghost has no contacts or effect on physics."""
    scene.ngeom = 0
    for name in ("upper_link", "lower_link", "pusher", "cube_geom"):
        geom_id = simulation.model.geom(name).id
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom, int(simulation.model.geom_type[geom_id]),
            simulation.model.geom_size[geom_id], predicted_data.geom_xpos[geom_id],
            predicted_data.geom_xmat[geom_id], np.array([0.95, 0.15, 0.8, 0.45]),
        )
        geom.category = mujoco.mjtCatBit.mjCAT_DECOR
        scene.ngeom += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--episode", default="best", help="best, worst, worst-still, or a zero-based replay index")
    parser.add_argument("--random", action="store_true", help="Generate fresh randomized episodes; automatically advance")
    parser.add_argument("--condition", choices=["any", "still"], default="any")
    parser.add_argument("--motion-threshold", type=float, default=0.002, help="Maximum actual cube excursion in meters for still cases")
    parser.add_argument("--seed", type=int, default=2026, help="First random episode seed; each new episode increments it")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="100 ms checkpoint for --random")
    parser.add_argument("--speed", type=float, default=0.5, help="Playback speed; default half speed")
    parser.add_argument("--seconds", type=float, help="Close after this many wall-clock seconds")
    args = parser.parse_args()
    if not np.isfinite(args.speed) or args.speed <= 0:
        parser.error("--speed must be positive and finite")
    if args.seconds is not None and (not np.isfinite(args.seconds) or args.seconds <= 0):
        parser.error("--seconds must be positive and finite")
    if not np.isfinite(args.motion_threshold) or args.motion_threshold < 0:
        parser.error("--motion-threshold must be nonnegative and finite")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    histories = []
    if args.random:
        import torch
        from .contact_learning import load_predictor
        torch.set_num_threads(2)
        predict = load_predictor(args.checkpoint, 5)

        def append_random():
            truth, prediction, info = generate_random_episode(args.seed + len(histories), predict, args.condition, args.motion_threshold)
            histories.append((truth, prediction))
            print("New randomized episode: " + json.dumps(info), flush=True)

        append_random()
        actual, predicted = (np.stack([pair[column] for pair in histories]) for column in (0, 1))
        episode, dt = 0, 0.1
        candidates = np.array([0])
    else:
        actual, predicted, dt = load_replay(args.run)
        condition = "still" if args.episode == "worst-still" else args.condition
        try:
            candidates = eligible_episodes(actual, condition, args.motion_threshold)
            local = choose_episode(actual[candidates], predicted[candidates], args.episode, args.motion_threshold) if args.episode in ("best", "worst", "worst-still") else None
            episode = int(candidates[local]) if local is not None else choose_episode(actual, predicted, args.episode)
            if episode not in candidates:
                raise ValueError("Requested episode does not match --condition")
        except ValueError as error:
            parser.error(str(error))
    # Import GUI components only when launching, so mapping tests stay headless.
    import mujoco.viewer

    simulation = ContactSimulation()
    ghost = mujoco.MjData(simulation.model)
    keys = SimpleQueue()
    duration = (actual.shape[1] - 1) * dt
    playback = 0.0
    paused = False
    started = previous = time.perf_counter()
    next_log = started
    print("Recorded MuJoCo: original colors. Frozen-model prediction: translucent MAGENTA.")
    print("6/7 previous/next episode (7 generates another in random mode); 8 pause/resume; 9 restart.")
    print("Both start together; predictions never receive later simulator states.")
    print("Random mode automatically advances. Saved replay loops; --condition filters browsing.")
    try:
        with mujoco.viewer.launch_passive(simulation.model, simulation.data, key_callback=keys.put) as viewer:
            with viewer.lock():
                viewer.cam.lookat[:] = [0.3, 0, 0.04]
                viewer.cam.distance = 1.6
                viewer.cam.azimuth = 100
                viewer.cam.elevation = -55
            previous = time.perf_counter()
            while viewer.is_running():
                now = time.perf_counter()
                if args.seconds is not None and now - started >= args.seconds:
                    break
                if not paused:
                    playback += (now - previous) * args.speed
                previous = now
                advance = args.random and playback >= duration + 1.0
                if playback >= duration + 1.0:
                    playback = 0.0
                while True:
                    try:
                        key = keys.get_nowait()
                    except Empty:
                        break
                    if key in (ord("6"), ord("7")):
                        if args.random:
                            if key == ord("7"):
                                advance = True
                            else:
                                episode = max(0, episode - 1)
                                advance = False
                        else:
                            position = int(np.flatnonzero(candidates == episode)[0])
                            episode = int(candidates[(position + (1 if key == ord("7") else -1)) % len(candidates)])
                        playback = 0.0
                    elif key == ord("8"):
                        paused = not paused
                    elif key == ord("9"):
                        playback = 0.0
                        advance = False
                if advance:
                    episode += 1
                    if episode == len(histories):
                        append_random()
                        actual, predicted = (np.stack([pair[column] for pair in histories]) for column in (0, 1))
                    playback = 0.0
                    previous = time.perf_counter()
                display_time = min(playback, duration)
                truth = sample_state(actual[episode], display_time, dt)
                prediction = sample_state(predicted[episode], display_time, dt)
                with viewer.lock():
                    set_display_state(simulation, simulation.data, truth)
                    set_display_state(simulation, ghost, prediction)
                    simulation.data.time = display_time
                    draw_prediction(viewer.user_scn, simulation, ghost)
                viewer.sync()
                if now >= next_log:
                    error_cm = 100 * np.linalg.norm(truth[4:6] - prediction[4:6])
                    actual_mm = 1000 * np.linalg.norm(actual[episode, :, 4:6] - actual[episode, 0, 4:6], axis=-1).max()
                    predicted_mm = 1000 * np.linalg.norm(prediction[4:6] - actual[episode, 0, 4:6])
                    print(f"Episode {episode}/{len(actual)-1} | t={display_time:.2f}s | cube error={error_cm:.1f} cm | actual max={actual_mm:.2f} mm | predicted motion={predicted_mm:.1f} mm | {'paused' if paused else 'playing'}", flush=True)
                    next_log = now + 1
                time.sleep(1 / 60)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
