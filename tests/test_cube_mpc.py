import mujoco
import numpy as np
import pytest
import torch

from simple_arm.contact_learning import load_predictor
from simple_arm.cube_mpc import (CHECKPOINT, DATASET, CEMController, Config, ModelAdapter,
                                 fit_distribution, interpolate, rollout, trajectory_cost)
from simple_arm.cube_mpc_experiment import (execute_first, initialize, load_cases, metrics,
                                          evaluate, run_episode, sustained_success)
from simple_arm.cube_mpc_viewer import apply_key, draw


LIMITS = np.array([[-1.57, 1.57], [-2.1, 2.1]])


def state():
    value = torch.zeros(13)
    value[8] = 1
    return value


def dynamics(current, action):
    result = current.clone()
    result[..., 4:6] += action * .02
    return result


def controller(predict=dynamics, **settings):
    return CEMController(predict, LIMITS, LIMITS, Config(command_rate_limit=0., horizon=6, knots=3, candidates=40, iterations=3, scoring="original", **settings))


def test_defaults_and_configuration():
    assert Config(command_rate_limit=0., ).elites == 26
    for settings in ({"horizon": 0}, {"candidates": 1}, {"knots": 16}, {"elite_fraction": 0},
                     {"std_floor": 0}, {"retention": 1}, {"initial_std": float("nan")}, {"seed": -1}):
        with pytest.raises(ValueError):
            Config(command_rate_limit=0., **settings)


def test_tensor_recursive_feedback_and_interpolation():
    actions = torch.ones(2, 3, 2)
    predicted = rollout(dynamics, state(), actions)
    torch.testing.assert_close(predicted[0, :, 4], torch.tensor([.02, .04, .06]))
    knots = torch.tensor([[[0., 1.], [1., 2.], [2., 3.]]])
    torch.testing.assert_close(interpolate(knots, 5)[0, :, 0], torch.tensor([0., .5, 1., 1.5, 2.]))
    assert interpolate(knots[:, :1], 5).shape == (1, 5, 2)


def test_cost_units_and_previous_command_penalty():
    c = Config(command_rate_limit=0., horizon=2, knots=2, scoring="original")
    predicted = state().expand(1, 2, 13).clone()
    predicted[..., 4] = .1
    predicted[..., 2:4] = 2
    actions = torch.ones(1, 2, 2)
    costs, _, _ = trajectory_cost(predicted, actions, torch.zeros(2), torch.zeros(2), torch.tensor(LIMITS), c)
    # Distance 1 + terminal 5; command changes average .5; velocity MSE 4.
    assert costs.item() == pytest.approx(6 + .01 * .5 + .001 * 4)
    same, _, _ = trajectory_cost(predicted, actions, torch.zeros(2), torch.ones(2), torch.tensor(LIMITS), c)
    assert (costs - same).item() == pytest.approx(.005, abs=1e-6)
    predicted[..., 0] = 1.58
    bad, _, violation = trajectory_cost(predicted, actions, torch.zeros(2), torch.ones(2), torch.tensor(LIMITS), c)
    assert torch.isinf(bad).all() and violation.all()


def test_distribution_population_std_retention_and_floor():
    c = Config(command_rate_limit=0., )
    elite = torch.tensor([[[1., 2.]], [[3., 4.]]])
    mean, std = fit_distribution(torch.zeros(1, 2), torch.full((1, 2), .5), elite, c)
    torch.testing.assert_close(mean, torch.tensor([[1.6, 2.4]]))
    torch.testing.assert_close(std, torch.full((1, 2), .9))
    _, std = fit_distribution(mean, torch.zeros_like(std), elite[:1], c)
    assert (std == .05).all()


def test_neural_planning_uses_no_physics_and_dynamics_change_action(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Planner stepped MuJoCo")
    monkeypatch.setattr(mujoco, "mj_step", forbidden)
    plan = controller().plan(state(), [.1, 0], [0, 0])
    reverse = controller(lambda s, a: dynamics(s, -a)).plan(state(), [.1, 0], [0, 0])
    assert plan.actions[0, 0] > 0 and reverse.actions[0, 0] < 0
    np.testing.assert_allclose(plan.predicted[:, 4:6], .02 * np.cumsum(plan.actions, axis=0), atol=1e-7)
    assert plan.cost == min(i["best_cost"] for i in plan.diagnostics["iterations"])
    assert all(i["elite_count"] == 4 for i in plan.diagnostics["iterations"])
    assert np.all(plan.actions >= LIMITS[:, 0] + .15) and np.all(plan.actions <= LIMITS[:, 1] - .15)


def test_seed_reset_warm_start_and_fresh_spread():
    a, b = controller(), controller()
    first = a.plan(state(), [.1, 0], [0, 0])
    np.testing.assert_array_equal(first.actions, b.plan(state(), [.1, 0], [0, 0]).actions)
    expected = interpolate(torch.tensor(np.vstack([first.actions[1:], first.actions[-1:]]))[None], 3)[0]
    second = a.plan(state(), [.1, 0], first.actions[0])
    np.testing.assert_allclose(second.diagnostics["initial_mean"], expected.numpy())
    assert second.diagnostics["initial_std"] == .5
    a.reset()
    np.testing.assert_array_equal(first.actions, a.plan(state(), [.1, 0], [0, 0]).actions)


@pytest.mark.parametrize("kind", ["nan", "joint"])
def test_infeasible_fallback_never_clips_predicted_state(kind):
    def bad(s, a):
        result = s.clone()
        result[..., 0] = float("nan") if kind == "nan" else 100
        return result
    planner = controller(bad)
    plan = planner.plan(state(), [.1, 0], [.2, -.3])
    assert plan.diagnostics["fallback"] and plan.cost is None
    assert plan.diagnostics["rejected"] == 120
    np.testing.assert_allclose(plan.actions, np.tile([.2, -.3], (6, 1)))
    assert np.isnan(plan.predicted).all() and planner.best is None
    with pytest.raises(ValueError, match="Previous command"):
        planner.plan(state(), [0, 0], [2, 0])


def test_included_mean_and_previous_candidates():
    calls = []
    def spy(s, a):
        calls.append(a.clone())
        return dynamics(s, a)
    planner = controller(spy)
    previous = torch.tensor([.2, -.1])
    planner.plan(state(), [.1, 0], previous)
    for a in calls[:6]:
        torch.testing.assert_close(a[0], previous)
        torch.testing.assert_close(a[1], previous)


def test_elite_selection_and_best_from_earlier_iteration(monkeypatch):
    samples = iter([torch.tensor([[[0., 0.]], [[0., 0.]], [[1., 0.]], [[-1., 0.]]]), torch.zeros(4, 1, 2)])
    monkeypatch.setattr(torch, "randn", lambda *args, **kwargs: next(samples))
    c = Config(command_rate_limit=0., horizon=1, knots=1, candidates=4, iterations=2, elite_fraction=.25, scoring="original")
    result = CEMController(dynamics, LIMITS, LIMITS, c).plan(state(), [.01, 0], [0, 0])
    # First iteration's .5 action wins; second iteration samples only .4 and hold.
    np.testing.assert_allclose(result.actions, [[.5, 0]])
    np.testing.assert_allclose(result.diagnostics["final_mean"], [[.4, 0]], atol=1e-6)
    assert result.diagnostics["iterations"][1]["best_cost"] == result.cost


def test_existing_output_is_refused_before_loading(tmp_path):
    with pytest.raises(FileExistsError, match="Preserving"):
        evaluate("absent.pt", "absent.npz", tmp_path, Config(command_rate_limit=0., ))


def test_checkpoint_exact_transition_frozen_width_and_validation(tmp_path):
    adapter = ModelAdapter()
    assert adapter.model[0].out_features == 256
    assert all(not p.requires_grad for p in adapter.model.parameters())
    assert torch.get_num_threads() == 2
    initial, _ = load_cases(DATASET)[99]
    command = initial[:2] + .1
    with torch.inference_mode():
        actual = adapter(torch.tensor(initial, dtype=torch.float32), torch.tensor(command, dtype=torch.float32)).numpy()
    np.testing.assert_allclose(actual, load_predictor(CHECKPOINT, 5)(initial, command), atol=1e-6)
    for mutation in ("stride", "normalization", "state", "physics", "action"):
        checkpoint = torch.load(CHECKPOINT, weights_only=True)
        if mutation == "stride":
            checkpoint["stride"] = 1
        elif mutation == "normalization":
            checkpoint["input_scale"][0] = 0
        else:
            checkpoint["metadata"][{"state": "state", "physics": "physics_dt", "action": "action"}[mutation]] = "bad"
        path = tmp_path / f"{mutation}.pt"
        torch.save(checkpoint, path)
        with pytest.raises(ValueError):
            ModelAdapter(path)


def test_first_command_only_exact_interval_and_fresh_observation(monkeypatch):
    initial, target = load_cases(DATASET)[23]
    simulation = initialize(initial)
    start_time = simulation.data.time
    command = initial[:2] + [.1, -.1]
    count, original = [], simulation.step
    def spy(action):
        count.append(np.asarray(action).copy())
        return original(action)
    monkeypatch.setattr(simulation, "step", spy)
    observed, fine = execute_first(simulation, command)
    assert len(count) == 50
    np.testing.assert_array_equal(count, np.tile(command, (50, 1)))
    assert simulation.data.time - start_time == pytest.approx(.1)
    np.testing.assert_array_equal(observed, simulation.extract())
    assert not np.array_equal(initial, observed)
    np.testing.assert_array_equal(fine[-1], observed)


def test_identical_baseline_controller_initialization_and_no_action():
    initial, target = load_cases(DATASET)[23]
    c = Config(command_rate_limit=0., horizon=3, knots=2, candidates=8, iterations=1)
    baseline = run_episode(initial, target, dynamics, c, duration=.2, mode="no-action")
    controlled = run_episode(initial, target, dynamics, c, duration=.2)
    np.testing.assert_array_equal(baseline["states"][0], controlled["states"][0])
    np.testing.assert_allclose(baseline["commands"], np.tile(initial[:2], (2, 1)))
    assert np.max(np.abs(baseline["states"][:, 4:6] - initial[4:6])) < 1e-12
    report = metrics(controlled, target)
    assert len(report["first_step_comparisons"]) == 2


def test_sustained_tolerance_and_viewer_decorations():
    assert sustained_success(np.full(250, .02))
    assert not sustained_success(np.full(249, .01))
    distance = np.full(251, .01)
    distance[-250] = .021
    assert not sustained_success(distance)
    distance[0], distance[1] = .1, .01
    assert sustained_success(distance)
    target, changed, reset = apply_key(ord("6"), [0, 0])
    np.testing.assert_allclose(target, [-.02, 0])
    assert changed and not reset
    assert apply_key(257, target)[2]
    simulation = initialize(load_cases(DATASET)[23][0])
    before = simulation.extract().copy()
    scene = mujoco.MjvScene(simulation.model, maxgeom=100)
    predicted = np.tile(before, (15, 1))
    draw(scene, target, predicted)
    assert scene.ngeom == 17
    assert all(g.category == mujoco.mjtCatBit.mjCAT_DECOR for g in scene.geoms[:scene.ngeom])
    np.testing.assert_array_equal(before, simulation.extract())
