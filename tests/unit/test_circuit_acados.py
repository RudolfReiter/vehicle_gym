"""Optional integration test; the committed race data are checked separately in CI."""

import numpy as np
import pytest

pytest.importorskip("acados_template")
from vehiclegym.planner.race_mpc import RaceMPC  # noqa: E402
from vehiclegym.planner.references import FrenetReference  # noqa: E402
from vehiclegym.road.circuit import Circuit  # noqa: E402


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
