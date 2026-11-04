from pathlib import Path

import mujoco
import numpy as np


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "arm.xml"


class ArmSimulation:
    """Two joint position targets in radians; one action spans 20 ms."""

    def __init__(self, model_path=MODEL_PATH, frame_skip=10, *, model=None):
        if not isinstance(frame_skip, int) or frame_skip < 1:
            raise ValueError("frame_skip must be a positive integer")
        self.model = model if model is not None else mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.frame_skip = frame_skip
        self.joint_ids = [self.model.joint(name).id for name in ("shoulder", "elbow")]
        self.qpos_indices = self.model.jnt_qposadr[self.joint_ids]
        self.qvel_indices = self.model.jnt_dofadr[self.joint_ids]
        self.actuator_ids = [self.model.actuator(name).id for name in ("shoulder_servo", "elbow_servo")]
        self.action_limits = self.model.actuator_ctrlrange[self.actuator_ids].copy()
        self.cube_id = self.model.body("cube").id
        self.reset()

    @property
    def control_dt(self):
        return self.model.opt.timestep * self.frame_skip

    def reset(self):
        """Restore the XML initial state and return a fresh observation."""
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        return self.get_observation()

    def step(self, action):
        """Clip two finite position targets to actuator limits and advance physics."""
        targets = np.asarray(action, dtype=float)
        if targets.shape != (2,) or not np.all(np.isfinite(targets)):
            raise ValueError("action must contain two finite joint targets in radians")
        self.data.ctrl[self.actuator_ids] = np.clip(
            targets, self.action_limits[:, 0], self.action_limits[:, 1]
        )
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        mujoco.mj_forward(self.model, self.data)
        return self.get_observation()

    def get_observation(self):
        """Return independent arrays, with cube velocities in world coordinates."""
        spatial_velocity = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
            self.cube_id, spatial_velocity, 0,
        )
        return {
            "time": float(self.data.time),
            "joint_positions": self.data.qpos[self.qpos_indices].copy(),
            "joint_velocities": self.data.qvel[self.qvel_indices].copy(),
            "cube_position": self.data.xpos[self.cube_id].copy(),
            "cube_orientation": self.data.xquat[self.cube_id].copy(),
            "cube_linear_velocity": spatial_velocity[3:].copy(),
            "cube_angular_velocity": spatial_velocity[:3].copy(),
        }
