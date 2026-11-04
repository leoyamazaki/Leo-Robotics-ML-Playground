"""Gravity-supported cube with constrained roll/pitch and physical floor friction."""

import hashlib
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .simulation import ArmSimulation, MODEL_PATH


def model_xml():
    root = ET.fromstring(MODEL_PATH.read_text(encoding="utf-8"))
    cube = root.find(".//body[@name='cube']")
    cube.set("pos", "0 0 0.035")
    cube.remove(cube.find("freejoint"))
    for name, axis, kind in (("x", "1 0 0", "slide"), ("y", "0 1 0", "slide"), ("z", "0 0 1", "slide"), ("yaw", "0 0 1", "hinge")):
        ET.SubElement(cube, "joint", name=f"cube_{name}", axis=axis, type=kind, limited="false", damping="0", armature="0")
    return ET.tostring(root, encoding="unicode")


def link_lengths():
    root = ET.fromstring(MODEL_PATH.read_text(encoding="utf-8"))
    return [float(root.find(f".//geom[@name='{name}']").get("fromto").split()[3]) for name in ("upper_link", "lower_link")]


def model_hash():
    return hashlib.sha256(model_xml().encode()).hexdigest()


class ContactSimulation(ArmSimulation):
    def __init__(self):
        super().__init__(frame_skip=1, model=mujoco.MjModel.from_xml_string(model_xml()))
        self.cube_q = [self.model.jnt_qposadr[self.model.joint(f"cube_{name}").id] for name in ("x", "y", "z", "yaw")]
        self.cube_v = [self.model.jnt_dofadr[self.model.joint(f"cube_{name}").id] for name in ("x", "y", "z", "yaw")]
        self.cube_geom = self.model.geom("cube_geom").id
        self.arm_geoms = {self.model.geom(name).id for name in ("upper_link", "lower_link", "pusher")}

    def contact(self):
        return any(
            contact.dist <= 0 and contact.efc_address >= 0 and (
                (contact.geom1 == self.cube_geom and contact.geom2 in self.arm_geoms)
                or (contact.geom2 == self.cube_geom and contact.geom1 in self.arm_geoms)
            ) for contact in self.data.contact
        )

    def extract(self):
        observation = self.get_observation()
        yaw = self.data.qpos[self.cube_q[3]]
        return np.concatenate([
            observation["joint_positions"], observation["joint_velocities"],
            observation["cube_position"], [np.sin(yaw), np.cos(yaw)],
            observation["cube_linear_velocity"], [observation["cube_angular_velocity"][2]],
        ])

    def initialize(self, arm, cube_xy, yaw, arm_velocity=None, cube_velocity=None):
        arm = np.asarray(arm, dtype=float)
        cube_xy = np.asarray(cube_xy, dtype=float)
        if arm.shape != (2,) or cube_xy.shape != (2,) or not np.isfinite(np.r_[arm, cube_xy, yaw]).all():
            raise ValueError("Initial arm, cube position and yaw must be finite")
        if np.any(arm < self.action_limits[:, 0]) or np.any(arm > self.action_limits[:, 1]):
            raise ValueError("Initial arm outside joint limits")
        if cube_velocity is not None:
            cube_velocity = np.asarray(cube_velocity, dtype=float)
            if cube_velocity.shape != (3,) or not np.isfinite(cube_velocity).all():
                raise ValueError("Cube velocity must contain finite vx, vy, yaw rate")
        if arm_velocity is not None:
            arm_velocity = np.asarray(arm_velocity, dtype=float)
            if arm_velocity.shape != (2,) or not np.isfinite(arm_velocity).all():
                raise ValueError("Arm velocity must contain two finite rates")
        self.reset()
        self.data.qpos[self.qpos_indices] = arm
        self.data.qpos[self.cube_q] = [*cube_xy, 0, yaw]
        self.data.ctrl[:] = arm
        mujoco.mj_forward(self.model, self.data)
        if self.contact():
            raise ValueError("Initial arm/cube contact before settling")
        for _ in range(100):
            self.step(arm)
        if self.contact():
            raise ValueError("Initial arm/cube contact after settling")
        if arm_velocity is not None:
            self.data.qvel[self.qvel_indices] = arm_velocity
        if cube_velocity is not None:
            self.data.qvel[np.asarray(self.cube_v)[[0, 1, 3]]] = cube_velocity
        mujoco.mj_forward(self.model, self.data)
        return self.extract()
