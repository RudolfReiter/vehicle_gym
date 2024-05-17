from dataclasses import dataclass
import numpy as np
import json
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class BaseModelParameters:
    """ Parameters for all vehicle models"""
    length_front: float = 1.6
    length_rear: float = 1.4
    chassis_width: float = 1.9  # VW Passat values
    chassis_length: float = 4  # VW Passat values
    mass: float = 1160
    maximum_velocity: float = 60  # Maximum velocity of the ego vehicle
    maximum_steering_rate: float = 0.39
    maximum_steering_angle: float = 0.35
    roll_force_coeff: float = 0.015  # https://www.e31.net/luftwiderstand.html
    drag_force_coeff: float = 0.29
    drag_area_head: float = 2.07

    @property
    def length(self):
        return self.length_front + self.length_rear
