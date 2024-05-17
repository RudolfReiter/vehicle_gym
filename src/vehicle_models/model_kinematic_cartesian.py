import casadi as cs
from vehicle_models.model_kinematic import (KinematicVehicleModel,
                                            KinematicModelParameters)
from acados_template import AcadosModel


class KinematicModelCartesian(KinematicVehicleModel):
    """ Base class for kinematic vehicle models with 5 states """

    def __init__(self, params: KinematicModelParameters):
        super().__init__(params)
        self.name = "fiveStateKinCart"

        self.state_x = self.x_states[0]
        self.state_y = self.x_states[1]
        self.state_theta = self.x_states[2]

        length = params.length_rear + params.length_front

        # Simple vehicle models with reference points:
        # https://www.coursera.org/lecture/intro-self-driving-cars/lesson-2-the-kinematicbicycle-model-Bi8yE
        beta = cs.arctan2(self.lr * cs.tan(self.state_delta), (self.lr + self.lf))
        self.rhs[0] = self.state_v * cs.cos(self.state_theta + beta)
        self.rhs[1] = self.state_v * cs.sin(self.state_theta + beta)
        self.rhs[2] = self.state_v / length * cs.tan(self.state_delta) * cs.cos(beta)

        self.post_init()


class KinematicModelCartesianSmall(KinematicVehicleModel):
    """ Base class for kinematic vehicle models with 4 states """

    def __init__(self, params: KinematicModelParameters):
        super().__init__(params, n_states=4)
        self.name = "fourStateKinCart"
        self.state_x = self.x_states[0]
        self.state_y = self.x_states[1]
        self.state_theta = self.x_states[2]
        length = self.lf + self.lr
        beta = cs.arctan2(self.lr * cs.tan(self.control_delta), (self.lr + self.lf))

        self.rhs[0] = self.state_v * cs.cos(self.state_theta + beta)
        self.rhs[1] = self.state_v * cs.sin(self.state_theta + beta)
        self.rhs[2] = self.state_v / length * cs.tan(self.control_delta) * cs.cos(self.beta)

        self.post_init()
