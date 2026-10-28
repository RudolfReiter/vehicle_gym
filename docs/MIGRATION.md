# Migration and review notes

This update is based on upstream `09791bf`, on branch `improve-research-interface`. The separate original checkout is unchanged.

| Previous location/API | Current location/API | Compatibility |
|---|---|---|
| `planner.predicitor` | `planner.predictor` | Import shim retained |
| `utils.evalutation_tools` | `utils.evaluation_tools` | Import shim retained |
| `tests/planner_tests/` | `examples/planner/` | Executable examples, no longer collected as tests |
| `tests/simulator/` | `examples/simulator/` | Executable examples |
| `animation*.gif` | `docs/assets/legacy/` | Original media preserved |
| `requirements.txt` all dependencies | `pyproject.toml` optional groups | Old list in `requirements-legacy.txt` |

The original root-level exported classes remain lazily accessible. Access to an optional backend still requires its dependencies. Existing acados scripts retain their APIs; the portable planner is an explicit new API, not an acados replacement hidden behind an existing class.

`Planner.get_formatted_solution()` now returns a new dataclass instance with copied arrays per call. Previously it wrote fields on the dataclass class itself. Reference validation now checks every axis and finite values; enabling `check_sizes` raises `ValueError` for invalid output. Rectangle collision checks now include intersecting edges without enclosed vertices, and still treat touching as contact.

The new source uses typed configuration/reference dataclasses, explicit failure semantics, package resources, and modules with a single responsibility. Upstream code has not been reformatted wholesale. Tests cover the portable path and targeted regressions; full acados integration remains to be verified in its compiled environment.
