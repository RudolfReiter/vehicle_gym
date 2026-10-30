"""Optional integration test; the committed race data are checked separately in CI."""

import numpy as np
import pytest

pytest.importorskip("acados_template")
from vehiclegym.planner.race_mpc import RaceMPC  # noqa: E402
from vehiclegym.planner.references import FrenetReference  # noqa: E402
from vehiclegym.road.circuit import Circuit  # noqa: E402
from vehiclegym.planner.vehicle_limits import VehicleLimits  # noqa: E402


def test_cost_weights_change_the_optimized_horizon():
    track = Circuit.load("IMS")
    planner = RaceMPC(track)
    state = np.array(
        [1200.0, 0.0, 0.0, 25.0, np.arctan(3 * np.interp(1200, track.s, track.curvature))]
    )
    others = np.array([[1400 + 100 * i, 0, 0, 25, 0] for i in range(5)], float)
    weak = planner.plan(state, others, FrenetReference(speed=28, lateral=2, lateral_weight=0.2))
    planner.reset()
    strong = planner.plan(state, others, FrenetReference(speed=28, lateral=2, lateral_weight=20))
    assert weak.success and strong.success
    assert np.mean(strong.states[1:10, 1]) > np.mean(weak.states[1:10, 1]) + 0.1
    np.testing.assert_allclose(strong.states[0], state, atol=1e-6)


def test_cached_solver_keeps_independent_acceleration_and_speed_bounds():
    track = Circuit.load("Oval")
    limits = [
        VehicleLimits(speed_max=16, longitudinal_max=1.5),
        VehicleLimits(speed_max=28, longitudinal_max=5),
    ]
    planners = [RaceMPC(track, limits=limit) for limit in limits]
    state = np.array([0, 0, 0, 12, np.arctan(3 * track.curvature_at(0))])
    others = np.array([[200 + 20 * i, 0, 0, 12, 0] for i in range(5)], float)
    results = []
    for planner, limit in zip(planners, limits):
        plan = planner.plan(state, others, FrenetReference(speed=35))
        assert plan.success
        acceleration = np.array(
            [
                np.asarray(planner.model.f_ode(x, u)).ravel()[3]
                for x, u in zip(plan.states, plan.controls)
            ]
        )
        assert acceleration.max() <= limit.longitudinal_max + 0.01
        assert acceleration.min() >= limit.longitudinal_min - 0.01
        assert plan.states[:, 3].max() <= limit.speed_max + 0.05
        results.append(plan.states[:, 3].max())
        wrapped = np.asarray(planner.integrate(state, [1, 0], 0.2)).ravel()
        shifted = state.copy()
        shifted[0] += 12 * track.length
        later = np.asarray(planner.integrate(shifted, [1, 0], 0.2)).ravel()
        later[0] -= 12 * track.length
        np.testing.assert_allclose(later, wrapped, atol=1e-8)
    assert results[1] > results[0] + 2
