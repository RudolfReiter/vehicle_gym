import numpy as np
from vehicle_models.model_dynamic import DynamicVehicleModel, DynVehicleModelParameters
import casadi as cs
from vehicle_models.models_tire import TireModel


class DynamicModelFrenet(DynamicVehicleModel):
    """ dynamic frenet vehicle model
        https://arxiv.org/pdf/2003.04882.pdf
    """

    def __init__(self, params: DynVehicleModelParameters, s_grid: np.ndarray, p_kappa, tire_model: TireModel):
        super().__init__(params=params, tire_model=tire_model)

        self.number_grid = s_grid.shape[0]
        self.s_grid = s_grid
        self.p_kappa = p_kappa
        self.s2kappa = cs.interpolant("interp_s2kappa", "linear", [s_grid])

        self.state_s = self.x_states[0]
        self.state_n = self.x_states[1]
        self.state_alpha = self.x_states[2]

        sdot = self.state_vx * cs.cos(self.state_alpha) - self.state_vy * cs.sin(self.state_alpha) / \
               (1 - self.state_n * self.s2kappa(self.state_s, self.p_kappa))

        self.rhs_dyn[0] = sdot
        self.rhs_dyn[1] = self.state_vx * cs.sin(self.state_alpha) + self.state_vy * cs.cos(self.state_alpha)
        self.rhs_dyn[2] = self.state_omega - self.s2kappa(self.state_s, self.p_kappa) * sdot

        s_dot_kin = self.state_vx * cs.cos(self.state_alpha) / (1 - self.state_n * self.s2kappa(self.state_s))
        length = params.length_rear + params.length_front
        self.rhs_kin[0] = s_dot_kin
        self.rhs_kin[1] = self.state_vx * cs.sin(self.state_alpha)
        self.rhs_kin[2] = self.state_vx / length * cs.tan(self.control_delta) - s_dot_kin * self.s2kappa(self.state_s)

        self.post_init()
