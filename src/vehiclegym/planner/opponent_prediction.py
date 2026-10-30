"""Opponent trajectories supplied to the MPC, following ECC 2023 section IV-B.3.

Leading/alongside vehicles keep their measured speed and heading error. A car
fully behind the ego is assigned a full-braking trajectory. The latter is a
feasibility assumption about the follower's ability to yield, not a forecast of
its intended behavior. No future simulator states enter this computation.
"""

import numpy as np


def normalize_opponents(opponents):
    values = np.asarray(opponents, dtype=float)
    if values.size == 0:
        return np.empty((0, 5))
    if values.ndim != 2 or values.shape[1] not in (3, 5):
        raise ValueError("Opponents must have columns [s,n,v] or [s,n,alpha,v,delta]")
    if values.shape[1] == 3:
        expanded = np.zeros((len(values), 5))
        expanded[:, [0, 1, 3]] = values
        values = expanded
    if not np.isfinite(values).all():
        raise ValueError("Opponent states must be finite")
    return values


def predict_opponents(ego, opponents, time_grid, track, braking=8.0):
    """Return (opponent, horizon node, state) trajectories and assumption labels."""
    opponents = normalize_opponents(opponents)
    trajectories = np.repeat(opponents[:, None, :], len(time_grid), axis=1)
    behind = opponents[:, 0] < ego[0] - 4.6
    modes = tuple("follower_braking" if value else "constant_velocity_heading" for value in behind)
    acceleration = np.where(behind, -braking, 0.0)
    state = opponents.copy()

    def derivative(value):
        curvature = np.interp(value[:, 0], track.s, track.curvature)
        return np.column_stack(
            (
                value[:, 3] * np.cos(value[:, 2]) / (1 - value[:, 1] * curvature),
                value[:, 3] * np.sin(value[:, 2]),
                np.zeros(len(value)),
                acceleration,
                np.zeros(len(value)),
            )
        )

    for k, interval in enumerate(np.diff(time_grid), start=1):
        # RK4 with each braking trajectory split exactly at standstill.
        dt = np.where(behind, np.minimum(interval, state[:, 3] / braking), interval)[:, None]
        k1 = derivative(state)
        k2 = derivative(state + dt * k1 / 2)
        k3 = derivative(state + dt * k2 / 2)
        k4 = derivative(state + dt * k3)
        state += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        state[:, 3] = np.maximum(0.0, state[:, 3])
        state[:, 1] = np.clip(state[:, 1], -track.half_width + 1.1, track.half_width - 1.1)
        trajectories[:, k] = state
    return trajectories, modes
