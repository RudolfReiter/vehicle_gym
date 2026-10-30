"""Per-car racing envelopes in SI units, separate from racing objectives."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class VehicleLimits:
    speed_min: float = 0.0
    speed_max: float = 32.0
    longitudinal_min: float = -8.0
    longitudinal_max: float = 5.0
    lateral_min: float = -9.0
    lateral_max: float = 9.0

    def __post_init__(self):
        values = list(self.__dict__.values())
        if not np.isfinite(values).all():
            raise ValueError("Vehicle limits must be finite")
        if not 0 <= self.speed_min < self.speed_max:
            raise ValueError("Require 0 <= speed_min < speed_max")
        for axis in ["longitudinal", "lateral"]:
            if not getattr(self, axis + "_min") < 0 < getattr(self, axis + "_max"):
                raise ValueError("Acceleration intervals must contain zero")


def racing_limits():
    """Mixed straight-line, braking and cornering capabilities; no maneuver scripts."""
    return [
        VehicleLimits(0, 32, -8.0, 5.0, -9.0, 9.0),
        VehicleLimits(0, 25, -5.0, 2.8, -5.5, 5.5),
        VehicleLimits(2, 29, -6.5, 4.0, -7.0, 7.5),
        VehicleLimits(0, 30, -9.0, 3.5, -8.0, 8.0),
        VehicleLimits(4, 36, -10.0, 6.0, -11.0, 10.5),
        VehicleLimits(2, 34, -7.5, 5.5, -9.5, 10.0),
    ]
