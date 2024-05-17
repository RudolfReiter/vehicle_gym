from vehicle_models.model_dynamic import DynamicVehicleModel, DynVehicleModelParameters, TireModel
import casadi as cs


class DynamicModelCartesian(DynamicVehicleModel):
    """ dynamic cartesian vehicle model
        https://arxiv.org/abs/1711.07300
        Ionescu_Jonsson_2019_Design Trade-offs in Optimisation Based Trajectory Planning for Autonomous.pdf
    """

    def __init__(self, params: DynVehicleModelParameters, tire_model: TireModel):
        super().__init__(params=params, tire_model=tire_model)

        self.state_x = self.x_states[0]
        self.state_y = self.x_states[1]
        self.state_theta = self.x_states[2]

        self.rhs_dyn[0] = self.state_vx * cs.cos(self.state_phi) - self.state_vy * cs.sin(self.state_phi)
        self.rhs_dyn[1] = self.state_vx * cs.sin(self.state_phi) + self.state_vy * cs.cos(self.state_phi)
        self.rhs_dyn[2] = self.state_omega
        length = params.length_rear + params.length_front
        beta_kin = cs.arctan2(params.length_rear * cs.tan(self.control_delta), length)

        self.rhs_kin[0] = self.state_vx * cs.cos(self.state_phi + beta_kin)
        self.rhs_kin[1] = self.state_vx * cs.sin(self.state_phi + beta_kin)
        self.rhs_kin[2] = self.state_vx / length * cs.tan(self.control_delta) * cs.cos(beta_kin)

        self.post_init()
