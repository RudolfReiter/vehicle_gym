"""Physical reference parameters shared by planners and policy wrappers."""

from dataclasses import dataclass
import numpy as np
from vehiclegym.planner.planner_reference_interfaces import ReferenceValues


@dataclass(frozen=True)
class FrenetReference:
    speed: float = 30.0
    lateral: float = 0.0
    speed_weight: float = 0.5
    lateral_weight: float = 1.0

    def __post_init__(self):
        values = [self.speed, self.lateral, self.speed_weight, self.lateral_weight]
        if not np.isfinite(values).all() or self.speed <= 0:
            raise ValueError("References must be finite and speed must be positive")
        if min(self.speed_weight, self.lateral_weight) < 0:
            raise ValueError("Cost weights must be nonnegative")

    def expand(self, state, time_grid):
        """Return the original vehicle_gym reference-matrix representation."""
        state = np.asarray(state, dtype=float)
        time_grid = np.asarray(time_grid, dtype=float)
        if state.shape != (5,) or not np.isfinite(state).all():
            raise ValueError("State must contain five finite values")
        if (
            time_grid.ndim != 1
            or len(time_grid) < 2
            or not np.isfinite(time_grid).all()
            or time_grid[0] != 0
            or np.any(np.diff(time_grid) <= 0)
        ):
            raise ValueError("Time grid must be finite, start at zero, and increase")
        q = np.diag([0.015, self.lateral_weight, 2.0, self.speed_weight, 2.0])
        ref = np.zeros((len(time_grid), 5))
        ref[:, 0] = state[0] + self.speed * time_grid
        ref[:, 1] = self.lateral
        ref[:, 3] = self.speed
        return ReferenceValues(
            np.repeat(q[None], len(time_grid) - 1, axis=0),
            q.copy(),
            np.repeat(np.diag([1e-8, 0.4])[None], len(time_grid) - 1, axis=0),
            ref,
            np.zeros((len(time_grid) - 1, 2)),
        )
