"""Vehicle planning, with optional backends loaded only when requested."""

from importlib import import_module

__version__ = "1.2.0"
_EXPORTS = {
    "PlannerOptions": "vehiclegym.planner.trajectory_planner_base",
    "PlanningDataContainer": "vehiclegym.planner.trajectory_planner_base",
    "VehiclePlannerAcados20221": "vehiclegym.planner.trajectory_planner_acados_20221",
    "AnimationParameters": "vehiclegym.animation.animator",
    "Animator": "vehiclegym.animation.animator",
    "Nmpc20224": "vehiclegym.controller.nmpc_20224",
    "NmpcOptions20224": "vehiclegym.controller.nmpc_20224",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name]), name)
    globals()[name] = value
    return value
