"""Checkpoint architecture and feature contracts shared by training and controllers."""

import numpy as np
import torch
from torch import nn


RESIDUAL_ARCHITECTURE = {"type": "residual_mlp", "width": 768, "blocks": 4}
INPUT_FLOOR = [0.1, 0.1, 0.5, 0.5, 0.03, 0.03, 0.01, 0.2, 0.2, 0.05, 0.05, 0.05, 0.2, 0.1, 0.1]
TARGET_FLOOR = [0.01, 0.01, 0.1, 0.1, 0.001, 0.001, 0.001, 0.01, 0.01, 0.01, 0.01, 0.05]
STATE_SCALES = [1, 1, 3, 3, .1, .1, .01, .5, .5, .5, .5, .1, 2]


class ResidualBlock(nn.Module):
    def __init__(self, width=768):
        super().__init__()
        self.linear1 = nn.Linear(width, width)
        self.linear2 = nn.Linear(width, width)

    def forward(self, x):
        return torch.nn.functional.silu(x + self.linear2(torch.nn.functional.silu(self.linear1(x))))


class ResidualMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(15, 768)
        self.blocks = nn.Sequential(*(ResidualBlock() for _ in range(4)))
        self.output = nn.Linear(768, 12)

    def forward(self, x):
        return self.output(self.blocks(torch.nn.functional.silu(self.input(x))))


def coordinate_frame(checkpoint):
    frame = checkpoint.get("input_coordinate_frame", "tip_relative")
    if frame not in ("tip_relative", "base_relative"):
        raise ValueError(f"Unknown input coordinate frame: {frame!r}")
    return frame


def model_from_checkpoint(checkpoint):
    """Missing configuration retains the historical three-layer plain MLP."""
    coordinate_frame(checkpoint)
    architecture = checkpoint.get("architecture", {"type": "plain_mlp", "width": checkpoint.get("hidden_width", 128), "layers": 3})
    if architecture == RESIDUAL_ARCHITECTURE:
        return ResidualMLP()
    if not isinstance(architecture, dict) or set(architecture) != {"type", "width", "layers"} or architecture["type"] != "plain_mlp" or architecture["layers"] != 3:
        raise ValueError(f"Unknown architecture configuration: {architecture!r}")
    return plain_model(architecture["width"])


def plain_model(width=128):
    if type(width) is not int or width < 1:
        raise ValueError("hidden_width must be a positive integer")
    return nn.Sequential(nn.Linear(15, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(),
                         nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 12))


def state_features(states, actions, lengths, frame="tip_relative"):
    coordinate_frame({"input_coordinate_frame": frame})
    tensor = isinstance(states, torch.Tensor)
    if frame == "base_relative":
        # Fixed base is world XY (0, 0); all other components are unchanged.
        return torch.cat([states, actions], -1) if tensor else np.concatenate([states, actions], -1)
    cos, sin = (torch.cos, torch.sin) if tensor else (np.cos, np.sin)
    angle, total = states[..., 0], states[..., 0] + states[..., 1]
    stack = torch.stack if tensor else np.stack
    tip = stack([lengths[0] * cos(angle) + lengths[1] * cos(total),
                 lengths[0] * sin(angle) + lengths[1] * sin(total)], -1)
    parts = [states[..., :4], states[..., 4:6] - tip, states[..., 6:], actions]
    return torch.cat(parts, -1) if tensor else np.concatenate(parts, -1)


def checkpoint_features(states, actions, checkpoint):
    return state_features(states, actions, checkpoint["metadata"]["lengths"], coordinate_frame(checkpoint))
