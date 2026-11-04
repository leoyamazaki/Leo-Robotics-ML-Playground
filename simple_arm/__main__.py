import argparse
from queue import Empty, SimpleQueue
import time

import glfw
import mujoco.viewer
import numpy as np

from .simulation import ArmSimulation


def main():
    parser = argparse.ArgumentParser(description="Interactive two-joint MuJoCo arm")
    parser.add_argument("--seconds", type=float, help="Close automatically after this many wall-clock seconds")
    args = parser.parse_args()
    if args.seconds is not None and (not np.isfinite(args.seconds) or args.seconds <= 0):
        parser.error("--seconds must be positive and finite")
    simulation = ArmSimulation()
    commands = SimpleQueue()
    targets = np.zeros(2)
    print("Focus viewer: 6/7 shoulder -/+; 8/9 elbow -/+; Enter reset; close window to exit.")
    started = time.perf_counter()
    next_log = started
    deadline = started
    try:
        with mujoco.viewer.launch_passive(
            simulation.model, simulation.data, key_callback=commands.put
        ) as viewer:
            with viewer.lock():
                viewer.cam.lookat[:] = [0.25, 0, 0.05]
                viewer.cam.distance = 1.4
                viewer.cam.azimuth = 120
                viewer.cam.elevation = -40
            while viewer.is_running():
                if args.seconds is not None and time.perf_counter() - started >= args.seconds:
                    break
                with viewer.lock():
                    while True:
                        try:
                            key = commands.get_nowait()
                        except Empty:
                            break
                        if key == glfw.KEY_ENTER:
                            simulation.reset()
                            targets[:] = 0
                        elif key in (ord("6"), ord("7")):
                            targets[0] += 0.1 if key == ord("7") else -0.1
                        elif key in (ord("8"), ord("9")):
                            targets[1] += 0.1 if key == ord("9") else -0.1
                    targets[:] = np.clip(targets, simulation.action_limits[:, 0], simulation.action_limits[:, 1])
                    observation = simulation.step(targets)
                viewer.sync()
                now = time.perf_counter()
                if now >= next_log:
                    print(f"t={observation['time']:.2f}s joint(rad)={observation['joint_positions'].round(3)} cube(m)={observation['cube_position'].round(3)}", flush=True)
                    next_log = now + 1
                deadline += simulation.control_dt
                time.sleep(max(0, deadline - time.perf_counter()))
                if time.perf_counter() - deadline > 0.1:
                    deadline = time.perf_counter()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
