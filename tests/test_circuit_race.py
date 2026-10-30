"""Protocol tests do not require a locally compiled acados installation."""

from types import SimpleNamespace
import numpy as np
from vehiclegym.road.circuit import Circuit
from vehiclegym.simulator.circuit_race import CircuitRace
from vehiclegym.planner.references import FrenetReference
from vehiclegym.planner.opponent_prediction import predict_opponents
from vehiclegym.planner.vehicle_limits import VehicleLimits, racing_limits
import pytest


class RecordingPlanner:
    def __init__(self, track, config, limits=None, opponent_limits=None):
        self.config = config
        self.calls = []

    def reset(self):
        self.calls = []

    def plan(self, state, opponents, reference):
        self.calls.append((state.copy(), opponents.copy(), reference))
        return SimpleNamespace(
            success=True,
            controls=np.zeros((self.config.intervals, 2)),
            opponent_modes=tuple("constant_velocity_heading" for _ in opponents),
        )

    def integrate(self, state, control, dt):
        result = state.copy()
        result[0] += dt * state[3]
        return result


def test_periodic_track_has_real_corners_and_continuous_seam():
    track = Circuit.load("Sepang")
    assert 5400 < track.length < 5600
    assert np.max(abs(track.base_curvature)) > 0.03
    np.testing.assert_allclose(
        track.position([-0.01, 0, 0.01]),
        track.position(np.array([-0.01, 0, 0.01]) + track.length),
        atol=1e-8,
    )
    assert np.linalg.norm(track.position(-0.01) - track.position(0.01)) < 0.021


def test_each_car_solves_from_the_same_snapshot_and_only_first_control_is_applied():
    world = CircuitRace(planner_factory=RecordingPlanner)
    world.reset(21)
    before = world.states.copy()
    _, _, _, _, info = world.step()
    for i, planner in enumerate(world.planners):
        assert len(planner.calls) == 1
        np.testing.assert_array_equal(planner.calls[0][0], before[i])
        np.testing.assert_array_equal(planner.calls[0][1], np.delete(before, i, axis=0))
    np.testing.assert_allclose(world.states[:, 0], before[:, 0] + world.config.dt * before[:, 3])
    assert info["control_sources"] == ["MPC"] * 6


def test_direct_sac_bypasses_only_ego_mpc():
    world = CircuitRace(planner_factory=RecordingPlanner)
    world.reset(21)
    _, _, _, _, info = world.step(direct_control=[1000, 0.02])
    assert not world.planners[0].calls
    assert all(len(p.calls) == 1 for p in world.planners[1:])
    assert info["plans"][0] is None
    assert info["references"][0] is None
    assert info["control_sources"] == ["SAC"] + ["MPC"] * 5


def test_hilepp_changes_only_ego_mpc_parameters():
    world = CircuitRace(planner_factory=RecordingPlanner)
    world.reset(21)
    reference = FrenetReference(speed=24, lateral=2, speed_weight=3, lateral_weight=4)
    _, _, _, _, info = world.step(reference=reference)
    assert world.planners[0].calls[0][2] == reference
    assert [p.calls[0][2] for p in world.planners[1:]] == world.profiles[1:]
    assert info["control_sources"] == ["HILEPP"] + ["MPC"] * 5


def test_after_passing_same_opponent_switches_to_braking_without_reverse_motion():
    track = Circuit.load("IMS")
    opponent = np.array([[100, 0, 0, 20, 0.0]])
    grid = np.arange(26) * 0.2
    ahead, modes = predict_opponents(
        np.array([90, 0, 0, 20, 0.0]), opponent, grid, track, braking=10
    )
    behind, changed = predict_opponents(
        np.array([106, 0, 0, 20, 0.0]), opponent, grid, track, braking=10
    )
    assert modes == ("constant_velocity_heading",)
    assert changed == ("follower_braking",)
    np.testing.assert_allclose(ahead[0, :, 3], 20)
    assert behind[0, -1, 3] == 0
    assert np.all(np.diff(behind[0, :, 0]) >= 0)
    np.testing.assert_allclose(behind[0, -1, 0], 120, atol=1e-7)


def test_periodic_opponent_positions_use_nearest_lap():
    track = Circuit.load("IMS")
    other = np.array([[track.length - 8, 0, 0, 20, 0.0]])
    np.testing.assert_allclose(track.nearby(np.array([2, 0, 0, 20, 0.0]), other)[0, 0], -8)
    assert other[0, 0] > 0


def test_oval_geometry_and_observations_remain_periodic_after_many_laps():
    world = CircuitRace("Oval", planner_factory=RecordingPlanner)
    obs = world.reset(33)
    assert 640 < world.track.length < 670
    np.testing.assert_allclose(world.track.base_xy[0], world.track.base_xy[-1], atol=1e-9)
    world.states[:, 0] += 12 * world.track.length
    np.testing.assert_allclose(world.observation(), obs, atol=1e-6)
    np.testing.assert_allclose(
        world.track.curvature_at([-10, 0, 100]),
        world.track.curvature_at(np.array([-10, 0, 100]) + 20 * world.track.length),
    )


def test_each_follower_prediction_uses_its_own_braking_capability():
    track = Circuit.load("Oval")
    cars = np.array([[-10, 0, 0, 20, 0], [-10, 0, 0, 20, 0]], float)
    predicted, modes = predict_opponents(
        np.zeros(5), cars, np.arange(31) * 0.2, track, braking=[4, 10]
    )
    assert modes == ("follower_braking",) * 2
    np.testing.assert_allclose(predicted[:, -1, 0], [40, 10], atol=1e-6)


def test_diverse_limits_and_invalid_envelopes():
    limits = racing_limits()
    assert len({v.speed_max for v in limits}) == 6
    assert len({v.longitudinal_max for v in limits}) == 6
    with pytest.raises(ValueError):
        VehicleLimits(speed_min=35, speed_max=30)
    with pytest.raises(ValueError):
        VehicleLimits(longitudinal_min=2)


def test_lap_pass_is_recorded_once_without_a_false_antipodal_pass():
    world = CircuitRace("Oval", planner_factory=RecordingPlanner)
    world.reset(1)
    world.states[:, 1] = [-4, -2, 0, 2, 4, 0]
    world.states[:, 2] = 0
    world.states[:, 0] = [world.track.length - 1, 0, 150, 220, 290, 360]
    world.states[:, 3] = [30, 0, 0, 0, 0, 0]
    world.pass_buckets = np.floor(
        (world.states[:, 0, None] - world.states[None, :, 0]) / world.track.length
    ).astype(int)
    world.step()
    passes = [e for e in world.events if e["kind"] == "pass"]
    assert len(passes) == 1 and passes[0]["lap_traffic"]
    assert (passes[0]["vehicle"], passes[0]["opponent"]) == (0, 1)
    world.step()
    assert sum(e["kind"] == "pass" for e in world.events) == 1
