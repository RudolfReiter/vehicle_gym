import numpy as np

from vehiclegym.planner.opponent_prediction import predict_opponents
from vehiclegym.planner.references import FrenetReference
from vehiclegym.road.track import Track
from vehiclegym.simulator.racing_world import RacingWorld


def test_role_dependent_assumptions_and_stopping_distance():
    opponents = np.array([[20, 0, 0, 25, 0], [-20, 0, 0, 10, 0], [-2, 3, 0, 12, 0]])
    times = np.arange(13) * 0.25
    trajectories, modes = predict_opponents([0, 0, 0, 30, 0], opponents, times, Track.synthetic(0))
    assert modes == ("constant_velocity_heading", "follower_braking", "constant_velocity_heading")
    np.testing.assert_array_equal(trajectories[:, 0], opponents)
    np.testing.assert_allclose(trajectories[0, :, 0], 20 + 25 * times)
    np.testing.assert_allclose(trajectories[1, 5:, 0], -20 + 10**2 / (2 * 8))
    np.testing.assert_array_equal(trajectories[1, 5:, 3], 0)
    assert np.all(np.diff(trajectories[1, :, 0]) >= 0)
    np.testing.assert_allclose(trajectories[2, :, 0], -2 + 12 * times)


def test_measured_heading_is_used_without_future_opponent_states():
    prediction, _ = predict_opponents(
        [0, 0, 0, 30, 0], [[20, 0, 0.05, 20, 0]], [0, 0.25, 0.5], Track.synthetic(0)
    )
    np.testing.assert_allclose(prediction[0, -1, :2], [20 + 10 * np.cos(0.05), 10 * np.sin(0.05)])


def test_every_step_replans_from_measurement_and_only_executes_first_control(monkeypatch):
    world = RacingWorld(max_steps=2)
    world.reset(10000, "blocking")
    solve = world.planner.plan
    measurements, controls = [], []

    def record(state, opponents, reference):
        measurements.append(state.copy())
        plan = solve(state, opponents, reference)
        assert plan.success
        controls.append(plan.controls[0].copy())
        # Poison the remaining horizon: executing it would change the next state.
        plan.controls[1:] = [8000, 0.39]
        return plan

    monkeypatch.setattr(world.planner, "plan", record)
    initial = world.state.copy()
    world.step(FrenetReference(speed=32))
    expected = initial
    for _ in range(5):
        expected = np.asarray(
            world.planner.integrate(expected, controls[0] / [1000, 1], 0.05)
        ).ravel()
    np.testing.assert_allclose(world.state, expected, atol=1e-10)
    world.step(FrenetReference(speed=30, lateral=2))
    assert len(measurements) == 2
    np.testing.assert_allclose(measurements[1], expected, atol=1e-10)
    assert not np.array_equal(measurements[0], measurements[1])


def test_blocking_opponents_are_faster_and_seek_a_passing_gap():
    world = RacingWorld()
    world.reset(10000, "blocking")
    assert np.all(world.speed_targets > world.config.speed_limit)
    assert np.all(world.opponents[:, 0] < world.state[0])
    world.state[1] = 0
    world.opponents[0] = [-10, 0, 0, 35, 0]
    world.lane_targets[0] = 0
    world._opponent_controls()
    assert abs(world.lane_targets[0]) > 3
