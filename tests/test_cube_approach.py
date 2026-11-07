from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
import torch
from simple_arm.cube_mpc import (Config, ModelAdapter, CEMController, goal_gate,
    score_components, trajectory_cost, tip_xy)
from simple_arm.cube_workspace import Workspace
from simple_arm.cube_mpc_experiment import load_cases, run_episode, initialize

CHECKPOINT = Path("runs/arm_cube_online/recursive/model.pt")
DATASET = Path("data/arm_cube_contact_boundary/independent.npz")


def test_approach_radius_and_observed_gate():
    w = Workspace()
    predicted = torch.zeros(4, 3, 13)
    predicted[..., 8] = 1
    tip = tip_xy(predicted, w)
    predicted[..., 4:6] = tip
    predicted[..., 4] += torch.tensor([.3, .2, .08, .04])[:, None]
    actions = torch.zeros(4, 3, 2)
    c = Config(scoring="approach")
    def cost(gate):
        return score_components(predicted, actions, torch.zeros(2), torch.zeros(2), c, w, gate)["approach"]
    values = cost(torch.tensor(1.))
    assert values[0] > values[1] > values[2]
    torch.testing.assert_close(values[2:], torch.zeros(2), atol=1e-12, rtol=0)
    observed = torch.zeros(13)
    gates = []
    for d in (0., .001, .01, .02, .1):
        observed[4] = d
        gates.append(float(goal_gate(observed, torch.zeros(2))))
    assert gates == sorted(gates) and gates[0] == 0
    assert gates[3] == pytest.approx(.5)
    torch.testing.assert_close(cost(torch.tensor(.5)), values * .5)


def test_original_exact_formula_and_zero_weight():
    torch.manual_seed(1)
    predicted = torch.rand(8, 15, 13) * .1
    actions = torch.rand(8, 15, 2)
    target, previous = torch.tensor([.2, .1]), torch.zeros(2)
    limits = torch.tensor([[-1., 1.], [-1., 1.]])
    c = Config()
    distance = (predicted[..., 4:6] - target).square().sum(-1) / .1**2
    changes = torch.diff(torch.cat([previous.expand(8, 1, 2), actions], 1), dim=1)
    expected = (distance.mean(1) + 5 * distance[:, -1] + .01 * changes.square().mean((1, 2))
                + .001 * predicted[..., 2:4].square().mean((1, 2)))
    original = trajectory_cost(predicted, actions, target, previous, limits, c)[0]
    assert torch.equal(original, expected)
    zero = trajectory_cost(predicted, actions, target, previous, limits,
        replace(c, scoring="approach", approach_weight=0), Workspace(limits), torch.tensor(1.))[0]
    assert torch.equal(original, zero)


def test_residual_frozen_execution_geometry_and_rate():
    adapter = ModelAdapter(CHECKPOINT)
    assert all(not p.requires_grad for p in adapter.model.parameters())
    assert adapter.checkpoint["input_coordinate_frame"] == "base_relative"
    initial, target = load_cases(DATASET, expected_seed=None)[23]
    simulation = initialize(initial)
    w = Workspace()
    # Compare the analytic tip to the actual endpoint of the pusher geom.
    pusher = simulation.model.geom("pusher").id
    np.testing.assert_allclose(tip_xy(torch.tensor(initial), w).numpy(), simulation.data.geom_xpos[pusher, :2], atol=1e-7)
    c = Config(scoring="approach")
    e = run_episode(initial, target, adapter, c, duration=.2)
    assert len(e["fine_states"]) == 100 and e["actual_contacts"].shape == (100,)
    assert np.isfinite(e["states"]).all()
    assert np.abs(np.diff(np.vstack([initial[:2], e["commands"]]), axis=0)).max() <= .02 + 1e-6
    assert all("approach" in d["score_components"] for d in e["diagnostics"])
    expected = goal_gate(torch.tensor(initial), torch.tensor(target))
    assert e["diagnostics"][0]["goal_gate"] == pytest.approx(float(expected))


def test_tip_approaches_stationary_cube():
    w = Workspace()
    p = torch.zeros(5, 2, 13)
    p[..., 0] = torch.tensor([.8, .5, .2, .1, 0.])[:, None]
    p[..., 4:6] = w.origin + torch.tensor([sum(w.lengths), 0.])
    cost = score_components(p, torch.zeros(5, 2, 2), torch.zeros(2), torch.zeros(2),
        Config(scoring="approach"), w, torch.tensor(1.))["approach"]
    assert torch.all(cost[:-2] > cost[1:-1])
    assert cost[-1] == 0


@pytest.mark.parametrize("kind", ["nan", "joint"])
def test_approach_rejects_infeasible_predictions(kind):
    def invalid(state, action):
        result = state.clone()
        result[..., 0] = float("nan") if kind == "nan" else 100
        return result
    limits = np.array([[-1., 1.], [-1., 1.]])
    c = Config(scoring="approach", candidates=4, iterations=1)
    plan = CEMController(invalid, limits, limits, c).plan(np.zeros(13), [.1, 0], [0, 0])
    assert plan.diagnostics["fallback"] and plan.cost is None
    assert not plan.actions.any()
    assert np.isnan(plan.predicted).all()


def test_gate_is_observed_once_per_plan(monkeypatch):
    import simple_arm.cube_mpc as module
    calls = []
    original = module.goal_gate
    def spy(observed, target):
        calls.append(observed.clone())
        return original(observed, target)
    monkeypatch.setattr(module, "goal_gate", spy)
    def dynamics(state, action):
        result = state.clone()
        result[..., 4] += .01
        return result
    state = np.zeros(13)
    state[4] = .2
    limits = np.array([[-1., 1.], [-1., 1.]])
    plan = CEMController(dynamics, limits, limits,
        Config(scoring="approach", horizon=3, knots=2, candidates=4, iterations=2)).plan(state, [0, 0], [0, 0])
    assert len(calls) == 1
    assert plan.diagnostics["goal_gate"] == pytest.approx(.2**2 / (.2**2 + .02**2))


def test_comparison_preserves_existing_output(tmp_path):
    from simple_arm.cube_approach_experiment import compare
    with pytest.raises(FileExistsError, match="Preserving"):
        compare("missing.pt", "missing.npz", tmp_path, Config())
