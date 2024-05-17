import numpy as np
from vehicle_models.model_dynamic import DynamicVehicleModel
import casadi as cs


class Simulator:
    def __init__(self, dynamic_vehicle_model: DynamicVehicleModel, steps : int = 1):

        self.model = dynamic_vehicle_model
        self.time_disc = cs.MX.sym('td')

        x_state_current = self.model.x_states
        td_current = self.time_disc/steps
        for i in range(steps):
            k1 = self.model.f_ode(x_state_current, self.model.u_controls)
            k2 = self.model.f_ode(x_state_current + td_current / 2 * k1, self.model.u_controls)
            k3 = self.model.f_ode(x_state_current + td_current / 2 * k2, self.model.u_controls)
            k4 = self.model.f_ode(x_state_current + td_current * k3, self.model.u_controls)
            x_state_current = x_state_current + td_current / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        self.x_next = x_state_current
        self.f_integrator = cs.Function('simulation_function_py',
                                        [self.model.x_states, self.model.u_controls, self.time_disc], [self.x_next])

    def simulate(self, x0: np.ndarray, u0: np.ndarray, t_sim: float) -> np.ndarray:
        x_next = self.f_integrator(x0, u0, t_sim).full()[:, 0]
        return x_next

    def generate_integrator(self, file_name="simulation_function.c"):
        # some more things we would like to know
        self.f_integrator_codegen = cs.Function('simulation_function',
                                                [self.model.x_states, self.model.u_controls, self.time_disc],
                                                [self.x_next, self.model.a_lon_cg, self.model.a_lat_cg, self.model.beta,
                                                 self.model.v_wheel_front, self.model.v_wheel_rear,
                                                 self.model.tire_model.sum_pressure_front,
                                                 self.model.tire_model.sum_pressure_rear])

        cg_options = {"with_header": True}
        self.f_integrator_codegen.generate(file_name, cg_options)
