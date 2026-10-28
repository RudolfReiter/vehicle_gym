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
    modes = []
    for index, initial in enumerate(opponents):
        behind = initial[0] < ego[0] - 4.6
        modes.append("follower_braking" if behind else "constant_velocity_heading")
        acceleration = -braking if behind else 0.0
        state = initial.copy()
        for k, interval in enumerate(np.diff(time_grid), start=1):
            # Integrate the curvilinear position with measured alpha held constant.
            # Split at standstill to avoid reverse motion during a braking assumption.
            moving_dt = min(interval, state[3] / braking) if behind else interval
            speed = max(0.0, state[3] + acceleration * moving_dt / 2)
            curvature = np.interp(state[0], track.s, track.curvature)
            state[0] += moving_dt * speed * np.cos(initial[2]) / (1 - state[1] * curvature)
            state[1] += moving_dt * speed * np.sin(initial[2])
            state[1] = np.clip(state[1], -track.half_width + 1.1, track.half_width - 1.1)
            state[3] = max(0.0, state[3] + acceleration * moving_dt)
            trajectories[index, k] = state
    return trajectories, tuple(modes)
