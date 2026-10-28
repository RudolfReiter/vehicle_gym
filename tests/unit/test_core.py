"""Regression coverage for import, packaging, geometry, and planner contracts."""

import subprocess
import sys
import numpy as np
import pytest

from vehiclegym.planner.planner_reference_interfaces import (
    CostInterface,
    CostInterfaceOptions,
    check_ref_matrix_sizes,
)
from vehiclegym.planner.references import FrenetReference
from vehiclegym.planner.trajectory_planner_base import Planner, PlanningData
from vehiclegym.road.track import Track
from vehiclegym.utils.evaluation_tools import check_collision_cartesian


def test_base_import_does_not_load_optional_backends():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, vehiclegym; "
            "assert 'acados_template' not in sys.modules; "
            "assert 'matplotlib' not in sys.modules",
        ],
        check=True,
    )


def test_planning_snapshots_are_independent():
    first, second = PlanningData(), PlanningData()
    first.x_opp_predicted_all.append("first")
    assert second.x_opp_predicted_all == []
    planner = Planner()
    planner.x_full = np.ones((5, 3))
    planner.time_grid = np.arange(3)
    planner.current_slacks = np.zeros(3)
    first = planner.get_formatted_solution()
    planner.x_full[:] = 2
    second = planner.get_formatted_solution(5)
    assert first is not second
    np.testing.assert_array_equal(first.x, 1)
    np.testing.assert_array_equal(second.x, 2)
    np.testing.assert_array_equal(second.t, [5, 6, 7])


def test_reference_shapes_and_failure_are_explicit():
    options = CostInterfaceOptions(
        q0=[1] * 5, qn=[1] * 5, r0=[1] * 2, time_grid=[0, 0.25, 0.5], check_sizes=True
    )
    reference = FrenetReference(28, 2).expand(np.zeros(5), options.time_grid)
    assert check_ref_matrix_sizes(reference, options)
    np.testing.assert_allclose(reference.x_ref[:, 0], [0, 7, 14])
    reference.Q = np.zeros(2)  # formerly indexed before validating dimensions
    assert not check_ref_matrix_sizes(reference, options)

    class BrokenInterface(CostInterface):
        def forward_(self, **kwargs):
            return reference

        def set_lateral_reference(self, n=None):
            pass

    with pytest.raises(ValueError, match="Reference matrices"):
        BrokenInterface(options).forward(np.zeros(5))


def test_packaged_track_from_other_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    track = Track.from_csv("Spielberg")
    assert track.s[-1] > 1000
    assert np.isfinite(track.curvature).all()
    np.testing.assert_allclose(track.position(track.s[0]), track.xy[0])


@pytest.mark.parametrize(
    "center,angle,expected",
    [
        ((0, 0), np.pi / 2, True),  # crossing edges; no vertex inside either rectangle
        ((10, 0), 0, False),
        ((6, 0), 0, True),  # touching counts as contact
    ],
)
def test_rectangle_collision(center, angle, expected):
    assert check_collision_cartesian((0, 0), (6, 1, 0), center, (6, 1, angle)) == expected


def test_legacy_misspelling_remains_compatible():
    from vehiclegym.utils.evalutation_tools import check_collision_cartesian as old

    assert old is check_collision_cartesian


@pytest.mark.parametrize("grid", [[0, np.nan], [0, 0], [[0, 1]]])
def test_invalid_reference_time_grid(grid):
    with pytest.raises(ValueError, match="Time grid"):
        FrenetReference().expand(np.zeros(5), grid)


def test_nonfinite_track_is_rejected():
    track = Track.synthetic()
    track.curvature[0] = np.nan
    with pytest.raises(ValueError):
        track.__post_init__()
