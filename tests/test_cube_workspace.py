from dataclasses import replace
import math

import mujoco
import numpy as np
import pytest
import torch

from simple_arm.cube_mpc import CEMController, Config, score_components, trajectory_cost
from simple_arm.cube_mpc_comparison import compare, workspace_metrics
from simple_arm.cube_workspace import Workspace


def predicted(xy=(.4, 0), velocity=(0, 0)):
    s = torch.zeros(1, 3, 13)
    s[..., 4:6] = torch.tensor(xy)
    s[..., 8] = 1
    s[..., 9:11] = torch.tensor(velocity)
    return s


def test_workspace_geometry_joint_limits_and_finite_object_sizes():
    w = Workspace()
    assert w.lengths == [.3, .25]
    assert w.contact_radius == pytest.approx(.057)
    assert w.outer_contact_bound == pytest.approx(.55 + .022 + math.hypot(.035, .035))
    inner = math.sqrt(.3**2 + .25**2 + 2 * .3 * .25 * math.cos(2.1))
    assert w.tip_distance(torch.tensor([[0., 0.]])).item() == pytest.approx(inner, abs=1e-6)
    assert w.excess(torch.tensor([[.62, 0.]])).item() == pytest.approx(.62 - .55 - .057, abs=1e-6)
    assert w.excess(torch.tensor([[.4, 0.]])).item() == 0
    # Same radius, different direction: joint restrictions exclude the rear point.
    assert w.excess(torch.tensor([[-.4, 0.]])).item() > .1
    limited = Workspace([[-.1, .1], [-.1, .1]])
    assert limited.excess(torch.tensor([[.4, 0.]])).item() > .08


def test_exact_arcs_against_dense_forward_kinematics_reference():
    w = Workspace()
    q1, q2 = torch.meshgrid(torch.linspace(*w.limits[0], 181), torch.linspace(*w.limits[1], 211), indexing="ij")
    tips = torch.stack([.3 * q1.cos() + .25 * (q1 + q2).cos(),
                        .3 * q1.sin() + .25 * (q1 + q2).sin()], -1).reshape(-1, 2)
    points = torch.rand(60, 2, generator=torch.Generator().manual_seed(9)) * 1.6 - .8
    exact = w.tip_distance(points)
    sampled = torch.cdist(points[None], tips[None])[0].min(-1).values
    # A grid can overestimate distance; its angular spacing bounds the gap.
    assert (exact <= sampled + 3e-5).all()
    assert (sampled - exact).max() < .006
    assert w.tip_distance(tips[::200]).max() < 1e-6


def test_velocity_is_direction_sensitive_and_allows_pushing_momentum():
    w, c = Workspace(), Config(command_rate_limit=0., scoring="workspace")
    benign = w.cost_components(predicted((.4, 0), (.2, 0)), c, .1)
    assert benign["excess_cube_speed"].item() == 0
    assert benign["escape_velocity"].item() == 0
    fast = w.cost_components(predicted((.4, 0), (.5, 0)), c, .1)
    assert fast["excess_cube_speed"].item() > 0
    outward = w.cost_components(predicted((.60, 0), (.2, 0)), c, .1)
    inward = w.cost_components(predicted((.60, 0), (-.2, 0)), c, .1)
    assert outward["escape_velocity"].item() > inward["escape_velocity"].item() == 0
    assert outward["terminal_workspace"].item() > inward["terminal_workspace"].item() == 0
    assert outward["terminal_cube_velocity"].item() > 0


def test_terminal_velocity_and_position_scored_and_breakdown_sums():
    w, c = Workspace(), Config(command_rate_limit=0., scoring="workspace")
    s = predicted()
    moving = s.clone()
    moving[:, -1, 9] = .2
    actions = torch.zeros(1, 3, 2)
    target = torch.tensor([.4, 0.])
    previous = torch.zeros(2)
    base = trajectory_cost(s, actions, target, previous, w.limits, c, w)[0]
    cost = trajectory_cost(moving, actions, target, previous, w.limits, c, w)[0]
    assert cost > base
    terms = score_components(moving, actions, target, previous, c, w)
    torch.testing.assert_close(cost, sum(terms.values()))
    farther = s.clone()
    farther[:, -1, 4] = .7
    assert trajectory_cost(farther, actions, target, previous, w.limits, c, w)[0] > cost


def test_original_exact_formula_and_zero_weights_recover_baseline():
    w = Workspace()
    c = Config(command_rate_limit=0., scoring="original")
    g = torch.Generator().manual_seed(7)
    s = torch.randn(10, 15, 13, generator=g) * .1
    actions = torch.randn(10, 15, 2, generator=g)
    target, previous = torch.tensor([.4, .1]), torch.tensor([.1, -.2])
    distance = (s[..., 4:6] - target).square().sum(-1) / .1**2
    changes = torch.diff(torch.cat([previous.expand(10, 1, 2), actions], 1), dim=1)
    old = c.distance_weight * distance.mean(1) + c.terminal_weight * distance[:, -1] + c.change_weight * changes.square().mean((1, 2)) + c.velocity_weight * s[..., 2:4].square().mean((1, 2))
    original = trajectory_cost(s, actions, target, previous, w.limits, c)[0]
    assert torch.equal(old, original)
    zero = replace(c, scoring="workspace", reachability_weight=0., cube_speed_weight=0.,
                   escape_velocity_weight=0., terminal_reachability_weight=0., terminal_cube_velocity_weight=0.)
    assert torch.equal(original, trajectory_cost(s, actions, target, previous, w.limits, zero, w)[0])


def test_workspace_planner_no_physics_and_selected_score_logging(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("physics used during planning")
    monkeypatch.setattr(mujoco, "mj_step", forbidden)
    def predict(s, a):
        result = s.clone()
        result[..., 4:6] += .02 * a
        result[..., 9:11] = .2 * a
        return result
    w = Workspace()
    c = Config(command_rate_limit=0., horizon=3, knots=2, candidates=16, iterations=2, scoring="workspace")
    planner = CEMController(predict, w.limits, w.limits, c)
    plan = planner.plan(predicted()[0, 0], [.43, 0], [0, 0])
    assert sum(plan.diagnostics["score_components"].values()) == pytest.approx(plan.cost, rel=1e-6)
    assert plan.diagnostics["scoring"] == "workspace"
    planner.reset()
    np.testing.assert_array_equal(plan.actions, planner.plan(predicted()[0, 0], [.43, 0], [0, 0]).actions)


def test_zero_added_weights_preserve_cem_sequences_and_warm_starts():
    w = Workspace()
    c = Config(command_rate_limit=0., horizon=3, knots=2, candidates=16, iterations=2)
    zero = replace(c, scoring="workspace", reachability_weight=0., cube_speed_weight=0.,
                   escape_velocity_weight=0., terminal_reachability_weight=0., terminal_cube_velocity_weight=0.)
    def dynamics(s, a):
        result = s.clone()
        result[..., 4:6] += .01 * a
        return result
    original, updated = [CEMController(dynamics, w.limits, w.limits, settings) for settings in (c, zero)]
    previous = np.zeros(2)
    for _ in range(2):
        a = original.plan(predicted()[0, 0], [.42, 0], previous)
        b = updated.plan(predicted()[0, 0], [.42, 0], previous)
        np.testing.assert_array_equal(a.actions, b.actions)
        np.testing.assert_array_equal(a.predicted, b.predicted)
        assert a.cost == b.cost
        previous = a.actions[0]


def test_actual_workspace_escape_metrics_and_comparison_guard(tmp_path):
    fine = np.zeros((4, 13))
    fine[:, 4] = [.4, .7, .4, .7]
    fine[1, 9] = 1.3
    report = workspace_metrics({"fine_states": fine}, Workspace())
    assert report["escaped"] and report["final_outside"]
    assert report["escape_events"] == 2 and report["outside_fraction"] == .5
    assert report["maximum_cube_speed_m_s"] == 1.3
    assert report["beyond_any_contact_outer_bound"]
    with pytest.raises(FileExistsError):
        compare("absent", "absent", tmp_path, Config(command_rate_limit=0., ))
    for settings in ({"scoring": "bad"}, {"free_cube_speed": 0}, {"reachability_weight": -1}):
        with pytest.raises(ValueError):
            Config(command_rate_limit=0., **settings)
