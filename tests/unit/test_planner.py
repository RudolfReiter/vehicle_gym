import numpy as np
import pytest

from vehiclegym.planner.portable import PredictivePlanner
from vehiclegym.planner.references import FrenetReference
from vehiclegym.road.track import Track
from vehiclegym.simulator.racing_world import RacingWorld


@pytest.fixture(scope="module")
def planner():
    return PredictivePlanner(Track.synthetic(bend=0))


def test_original_model_straight_motion_and_resistance(planner):
    state = np.array([0, 0, 0, 20, 0.0])
    derivative = np.asarray(planner.model.f_ode(state, [0, 0])).ravel()
    np.testing.assert_allclose(derivative[[0, 1, 2, 4]], [20, 0, 0, 0])
    assert derivative[3] < 0  # original rolling resistance and aerodynamic drag
    next_state = np.asarray(planner.integrate(state, [0, 0], 0.01)).ravel()
    np.testing.assert_allclose(next_state, state + 0.01 * derivative, atol=1e-4)


def test_feasible_plan_satisfies_dynamics_and_bounds(planner):
    planner.reset()
    plan = planner.plan([0, 0, 0, 22, 0], [[35, 0, 20]], FrenetReference(lateral=3))
    assert plan.success, plan.status
    assert plan.max_violation < 2e-4
    assert plan.max_obstacle_slack <= 0.0802
    assert np.max(abs(plan.controls[:, 1])) <= 0.3902
    for x, u, next_x in zip(plan.states, plan.controls, plan.states[1:]):
        predicted = np.asarray(planner.step(x, u / [1000, 1])).ravel()
        np.testing.assert_allclose(predicted, next_x, atol=2e-4)


def test_infeasible_plan_is_rejected(planner):
    plan = planner.plan([0, 10, 0, 22, 0], [], FrenetReference())
    assert not plan.success
    assert planner.previous is None


def test_seed_and_time_limit():
    world = RacingWorld(max_steps=1)
    obs = world.reset(seed=42, scenario="blocking")
    np.testing.assert_array_equal(obs, world.reset(seed=42, scenario="blocking"))
    _, _, terminated, truncated, info = world.step(FrenetReference())
    assert not terminated and truncated
    assert not info["solver_failed"]
    with pytest.raises(RuntimeError, match="reset"):
        world.step(FrenetReference())


def test_failed_solve_does_not_execute_control():
    world = RacingWorld()
    world.reset(seed=3)
    world.state[1] = 10
    before = world.state.copy()
    _, reward, terminated, truncated, info = world.step(FrenetReference())
    assert terminated and not truncated and info["solver_failed"]
    assert reward == -10
    np.testing.assert_array_equal(world.state, before)
