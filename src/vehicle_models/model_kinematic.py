from abc import ABC
from dataclasses import dataclass
from dataclasses_json import dataclass_json
import casadi as cs
import numpy as np
from acados_template import AcadosModel
from vehicle_models.base_parameters import BaseModelParameters
from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from vehiclegym.utils.acados_param_handler import ParamHandler


@dataclass_json
@dataclass
class KinematicModelParameters(BaseModelParameters):
    """ Parameters for kinematic vehicle models"""
    maximum_acceleration_force: float = 10e3
    maximum_deceleration_force: float = 15e3
    maximum_lateral_acc: float = 8  # Maximum lateral acceleration for both cars
    maximum_alpha: float = np.pi / 2 * 0.5
    safety_radius_side: float = 3.  # circular vehicle shapes
    comfort_radius: float = 8.0
    safety_radius_length: float = 5  # a different safety distance longitudinal
    safety_radius_boundary: float = 2


class KinematicVehicleModel(ABC):
    """ Base class for kinematic vehicle models with 4 or 5 states """

    def __init__(self, params: KinematicModelParameters,
                 param_handler: "ParamHandler",
                 n_states: int = 5,
                 add_states: int = 0):
        self.params_ = params
        self.rhs = None
        self.f_ode = None
        self.name = None
        self.param_handler = param_handler

        self.x_states = cs.MX.sym('x_states', n_states + add_states)
        self.u_controls = cs.MX.sym('u_controls', 2)

        self.state_v = self.x_states[3]
        self.control_Fx = self.u_controls[0]

        self.lr = params.length_rear
        self.lf = params.length_front
        length = self.lr + self.lf

        self.rhs = [0., 0., 0.]
        res_forces = 1 / 2 * params.drag_force_coeff * params.drag_area_head * 1.29 * self.state_v ** 2 + \
                     params.mass * params.roll_force_coeff * 9.81

        # distinguish delta or diff-delta input models
        if n_states == 5:
            self.state_delta = self.x_states[4]
            self.control_ddelta = self.u_controls[1]
            delta = self.state_delta
            self.rhs = self.rhs + \
                       [1 / params.mass * (self.control_Fx - res_forces),
                        self.control_ddelta]
        elif n_states == 4:
            self.control_delta = self.u_controls[1]
            delta = self.control_delta
            self.rhs = self.rhs + \
                       [1 / params.mass * (self.control_Fx - res_forces)]
        else:
            raise Exception()

        self.rhs = self.rhs + [0.] * add_states

        self.delta = delta
        self.beta = cs.arctan(self.lr / (self.lr + self.lf) * cs.tan(delta))

        # Lateral acceleration at rear wheel
        self.a_lat = self.state_v * self.state_v / length * cs.tan(delta)

        # Lateral acceleration at center of gravity (cg)
        self.a_lat_cg = self.control_Fx / params.mass * cs.sin(self.beta) + \
                        self.state_v * self.state_v / length * cs.cos(self.beta) * cs.tan(delta)

        self.f_a_lat_cg = cs.Function('f_a_lat', [self.x_states, self.u_controls], [self.a_lat_cg])
        self.f_a_lat = cs.Function('f_a_lat', [self.x_states], [self.a_lat])

        add_xdots = []
        for i in range(n_states + add_states):
            add_xdots.append(cs.MX.sym('x' + str(i) + '_dot'))

        self.xdot = cs.vertcat(*add_xdots)

    @property
    def n_x(self):
        return self.rhs.shape[0]

    def post_init(self):
        self.rhs = cs.vertcat(*self.rhs)
        self.f_ode = cs.Function('f_ode', [self.x_states, self.u_controls], [self.rhs], {"allow_free": True})

    def export4acados(self):
        model = AcadosModel()
        model.f_impl_expr = self.xdot - self.rhs
        model.f_expl_expr = self.rhs
        model.x = self.x_states
        model.xdot = self.xdot
        model.u = self.u_controls

        if hasattr(self, "z"):
            model.z = cs.vertcat(*self.z)

        model.p = self.param_handler.get_all_casadi()
        model.name = self.name

        if not hasattr(self, "state_s"):
            raise NotImplementedError
            # constraint_expr = cs.vertcat(self.a_lat_cg, self.state_x, self.state_y)
        elif hasattr(self, "state_s"):
            if hasattr(self, "p_x_opp") and hasattr(self, "p_kappa"):
                constraints_acc = cs.vertcat(self.a_lat_cg)
                constraints_boundary = cs.vertcat(*self.boundary_constr)
                constraints_obst = cs.vertcat(*self.safe_dist)
            else:
                raise NotImplementedError
                # constraint_expr = cs.vertcat(self.a_lat_cg, self.state_s, *self.boundary_constr)
        else:
            raise Exception()
        return model, constraints_acc, constraints_boundary, constraints_obst


