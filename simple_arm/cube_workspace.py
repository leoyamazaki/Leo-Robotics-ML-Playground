"""Planar kinematic reach, with no physics rollout or contact queries."""

import math
import xml.etree.ElementTree as ET

import torch

from .contact_simulation import link_lengths
from .simulation import MODEL_PATH


class Workspace:
    """Exact distance to the joint-limited 2R tip set, dilated for object size.

    The dilation uses the cube's inscribed disk plus the pusher radius: a
    conservative, yaw-independent contact-capable center workspace. This is
    kinematic reach only; it does not guarantee a useful pushing direction.
    """

    def __init__(self, joint_limits=None):
        root = ET.fromstring(MODEL_PATH.read_text(encoding="utf-8"))
        self.lengths = link_lengths()
        self.limits = (torch.tensor([list(map(float, root.find(f".//joint[@name='{name}']").get("range").split()))
                                     for name in ("shoulder", "elbow")]) if joint_limits is None
                       else torch.as_tensor(joint_limits, dtype=torch.float32).clone())
        # The current robot's two revolute joints share world Z and zero XY offsets
        # except for the first link. Keep the shoulder origin in world coordinates.
        self.origin = torch.tensor([sum(float(root.find(f".//body[@name='{name}']").get("pos").split()[i])
                                        for name in ("base", "upper_arm")) for i in (0, 1)])
        cube_half = list(map(float, root.find(".//geom[@name='cube_geom']").get("size").split()))
        pusher_radius = float(root.find(".//geom[@name='pusher']").get("size"))
        self.contact_radius = pusher_radius + min(cube_half[:2])
        self.outer_contact_bound = sum(self.lengths) + pusher_radius + math.hypot(*cube_half[:2])
        # Boundary of FK(rectangle) is contained in FK(rectangle edges) and
        # critical curves sin(elbow)=0. Each image is an analytic circular arc.
        first, second = self.lengths
        shoulder, elbow = self.limits.tolist()
        self.arcs = []
        for angle in shoulder:
            self.arcs.append(([first * math.cos(angle), first * math.sin(angle)], second,
                              angle + elbow[0], angle + elbow[1]))
        critical = [k * math.pi for k in range(math.ceil(elbow[0] / math.pi), math.floor(elbow[1] / math.pi) + 1)]
        for angle in (*elbow, *critical):
            x, y = first + second * math.cos(angle), second * math.sin(angle)
            offset = math.atan2(y, x)
            self.arcs.append(([0., 0.], math.hypot(x, y), shoulder[0] + offset, shoulder[1] + offset))

    @staticmethod
    def angle_in_range(angle, low, high):
        # A wrapped angle has a representative in this closed joint interval.
        return torch.ceil((low - angle) / (2 * math.pi)) <= torch.floor((high - angle) / (2 * math.pi))

    def tip_distance(self, xy):
        xy = xy - self.origin.to(xy)
        first, second = self.lengths
        limits = self.limits.to(xy)
        cosine = (xy.square().sum(-1) - first**2 - second**2) / (2 * first * second)
        possible_radius = (cosine >= -1) & (cosine <= 1)
        elbow = torch.acos(cosine.clamp(-1, 1))
        theta = torch.atan2(xy[..., 1], xy[..., 0])
        inside = torch.zeros_like(cosine, dtype=torch.bool)
        for sign in (-1, 1):
            q2 = sign * elbow
            q1 = theta - torch.atan2(second * torch.sin(q2), first + second * torch.cos(q2))
            inside |= (possible_radius & self.angle_in_range(q1, *limits[0])
                       & self.angle_in_range(q2, *limits[1]))
        distances = []
        for center, radius, low, high in self.arcs:
            relative = xy - xy.new_tensor(center)
            angle = torch.atan2(relative[..., 1], relative[..., 0])
            radial = (torch.linalg.vector_norm(relative, dim=-1) - radius).abs()
            endpoints = xy.new_tensor([[center[0] + radius * math.cos(a), center[1] + radius * math.sin(a)] for a in (low, high)])
            endpoint_distance = torch.linalg.vector_norm(xy[..., None, :] - endpoints, dim=-1).min(-1).values
            distances.append(torch.where(self.angle_in_range(angle, low, high), radial, endpoint_distance))
        distance = torch.stack(distances, -1).min(-1).values
        return torch.where(inside, torch.zeros_like(distance), distance)

    def excess(self, xy):
        """Meters beyond the contact-capable center workspace; zero inside."""
        return (self.tip_distance(xy) - self.contact_radius).clamp_min(0)

    def cost_components(self, predicted, config, dt):
        xy, velocity = predicted[..., 4:6], predicted[..., 9:11]
        excess = self.excess(xy)
        coasting_excess = self.excess(xy + dt * velocity)
        speed = torch.linalg.vector_norm(velocity, dim=-1)
        # Only outward boundary crossing or further escape counts here.
        escape_speed = (coasting_excess - excess).clamp_min(0) / dt
        return {
            "workspace": config.reachability_weight * (excess / .1).square().mean(1),
            "excess_cube_speed": config.cube_speed_weight * ((speed - config.free_cube_speed).clamp_min(0) / config.free_cube_speed).square().mean(1),
            "escape_velocity": config.escape_velocity_weight * (escape_speed / config.free_cube_speed).square().mean(1),
            "terminal_workspace": config.terminal_reachability_weight * (coasting_excess[:, -1] / .1).square(),
            "terminal_cube_velocity": config.terminal_cube_velocity_weight * (speed[:, -1] / config.free_cube_speed).square(),
        }
