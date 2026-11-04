"""Headless physical and interface checks; run with the project interpreter."""

import mujoco
import numpy as np

from simple_arm import ArmSimulation


def main():
    simulation = ArmSimulation()
    initial = simulation.reset()
    assert simulation.model.nu == 2
    falling = simulation.step([0, 0])
    assert falling["cube_position"][2] < initial["cube_position"][2]
    assert falling["cube_linear_velocity"][2] < 0
    ground_contact = False
    ground_id = simulation.model.geom("ground").id
    cube_id = simulation.model.geom("cube_geom").id
    for _ in range(100):
        settled = simulation.step([0, 0])
        ground_contact |= any(
            {contact.geom1, contact.geom2} == {ground_id, cube_id}
            for contact in simulation.data.contact
        )
    assert ground_contact
    assert abs(settled["cube_position"][2] - 0.035) < 0.002
    assert np.linalg.norm(settled["cube_linear_velocity"]) < 0.01
    baseline = settled["cube_position"].copy()
    arm_ids = {simulation.model.geom(name).id for name in ("upper_link", "lower_link", "pusher")}
    arm_contact = False
    for _ in range(150):
        pushed = simulation.step([0.65, 0])
        arm_contact |= any(
            (contact.geom1 == cube_id and contact.geom2 in arm_ids)
            or (contact.geom2 == cube_id and contact.geom1 in arm_ids)
            for contact in simulation.data.contact
        )
    displacement = np.linalg.norm(pushed["cube_position"][:2] - baseline[:2])
    assert arm_contact, "Arm must physically contact the cube"
    assert displacement > 0.02, f"Cube moved only {displacement} m"
    for _ in range(150):
        tracked = simulation.step([-0.4, 0.6])
    assert np.allclose(tracked["joint_positions"], [-0.4, 0.6], atol=0.03)
    for key, value in tracked.items():
        assert np.all(np.isfinite(value)), key
    assert tracked["joint_positions"].shape == (2,)
    assert tracked["joint_velocities"].shape == (2,)
    for key in ("cube_position", "cube_linear_velocity", "cube_angular_velocity"):
        assert tracked[key].shape == (3,)
    assert np.isclose(np.linalg.norm(tracked["cube_orientation"]), 1)
    tracked["joint_positions"][:] = 100
    assert np.max(np.abs(simulation.get_observation()["joint_positions"])) < 3
    simulation.step([100, -100])
    assert np.allclose(simulation.data.ctrl, [1.57, -2.1])
    for invalid in ([0], [0, np.nan], [0, 0, 0]):
        try:
            simulation.step(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid action accepted")
    reset = simulation.reset()
    assert reset["time"] == 0
    assert np.allclose(reset["cube_position"], initial["cube_position"])
    assert np.allclose(reset["joint_positions"], 0)
    assert np.allclose(reset["joint_velocities"], 0)
    print(f"PASS: MuJoCo {mujoco.__version__}; load, gravity, floor contact, both servos, reset, observations, action validation.")
    print(f"PASS: arm contact pushed cube {displacement:.3f} m.")


if __name__ == "__main__":
    main()
