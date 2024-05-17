from abc import ABC
from dataclasses import dataclass, field
import casadi as cs

from vehicle_models.base_parameters import BaseModelParameters
from vehicle_models.models_tire import TireModel, TireParameters, LinearTireParameters

'''
    Ionescu_Jonsson_2019_Design Trade-offs in Optimisation Based Trajectory Planning for Autonomous.pdf
    See https://arxiv.org/pdf/2003.04882.pdf, 
    https://arxiv.org/abs/1711.07300 
    for more details about the vehicle model
'''


@dataclass
class DriveParameters:
    c_rolling: float = 0  # 0.013
    c_air: float = 0.013  # TUM says: drag_coefficient = 0.75
    k_brake_r: float = 0.4  # part of braking force on rear wheel
    factor_pressure2force_rear: float = 0.005002 * 0.175 / (0.73009278 / 2) * 0.325  # out of strange simulink model
    factor_pressure2force_front: float = 0.005002 * 0.175 / (0.65212608 / 2) * 0.325


@dataclass
class DynVehicleModelParameters(BaseModelParameters):
    """ Parameters for dynamic vehicle models"""
    wheel_radius_rear: float = 0.73009278 / 2
    wheel_radius_front: float = 0.65212608 / 2
    inertia: float = 1160
    blend_factor: float = 10.  # kinematik and dynamic models get blended with
    blend_offset: float = 5
    f_drive_max: float = 7000
    f_brake_max: float = 20000
    # tanh(0.1*v_x) * dyn + (1-tanh(0.1*v_x)) *kin
    tires: TireParameters = field(default_factory=lambda: LinearTireParameters())
    drive: DriveParameters = field(default_factory=lambda: DriveParameters())


class DynamicVehicleModel(ABC):
    """ Base class for dynamic vehicle model """

    def __init__(self, params: DynVehicleModelParameters, tire_model: TireModel):
        self.params_ = params

        self.x_states = cs.MX.sym('x_states', 6)
        self.u_controls = cs.MX.sym('u_controls', 5)
        self.rhs = cs.MX.sym('rhs', 6)
        self.f_ode = None

        self.state_x = self.x_states[0]
        self.state_y = self.x_states[1]
        self.state_phi = self.x_states[2]
        self.state_vx = self.x_states[3]
        self.state_vy = self.x_states[4]
        self.state_omega = self.x_states[5]

        # The steering angle must always be controlled
        # The acceleration/deceleration can either be set by torques or the frame force
        self.control_force_tires = self.u_controls[0]

        # steering angle
        self.control_delta = self.u_controls[1]

        # torque forces
        self.control_torque_rear = self.u_controls[2]
        self.control_brake_pressure_front = self.u_controls[3]  # pressure for one tire
        self.control_brake_pressure_rear = self.u_controls[4]

        tire_model.set_variables(self.x_states, self.u_controls)
        Ffx, Frx, Ffy, Fry = tire_model.get_forces()
        self.tire_model = tire_model

        # normal tire loads
        Fz = params.mass * 9.81

        F_resistance_x = cs.tanh(self.state_vx) * (
                - Fz * params.drive.c_rolling) - params.drive.c_air * self.state_vx ** 2
        F_sum_x = F_resistance_x + 2 * Ffx + 2 * Frx + params.mass * self.state_vy * self.state_omega
        F_sum_y = 2 * Ffy + 2 * Fry - params.mass * self.state_vx * self.state_omega

        self.rhs_dyn = [
            0,
            0,
            0,
            1 / params.mass * F_sum_x,
            1 / params.mass * F_sum_y,
            1 / params.inertia * (2 * Ffy * params.length_front - 2 * Fry * params.length_rear)]

        # Lateral acceleration at center of gravity (cg)
        length = params.length_rear + params.length_front
        self.a_lat_cg = 1 / params.mass * F_sum_y
        self.a_lat_cg = self.state_vx * self.state_vx / length * cs.tan(self.control_delta)  # TODO: Totally not sure about that
        self.a_lon_cg = 1 / params.mass * F_sum_x
        self.f_a_lat_cg = cs.Function('f_a_lat', [self.x_states, self.u_controls], [self.a_lat_cg])
        self.f_a_lon_cg = cs.Function('f_a_lon', [self.x_states, self.u_controls], [self.a_lon_cg])

        # Slip angle
        self.beta = cs.arctan2(self.state_vy, self.state_vx)

        # Tire speeds approximately
        self.v_wheel_front = self.state_vx / self.params_.wheel_radius_front
        self.v_wheel_rear = self.state_vx / self.params_.wheel_radius_rear

        # Kinematic model for blending zero velocities
        F_sum_kin_x = (self.tire_model.force_long_rear_single_tire + self.tire_model.force_long_front_single_tire) * 2
        self.rhs_kin = [0,
                        0,
                        0,
                        F_sum_kin_x / params.mass,  # + F_resistance_x,
                        0,
                        0]

    def post_init(self):
        rhs_kin = cs.vertcat(*self.rhs_kin)
        rhs_dyn = cs.vertcat(*self.rhs_dyn)
        relation_dyn = (1 + cs.tanh((self.state_vx - self.params_.blend_offset) * self.params_.blend_factor)) * 0.5
        # x=np.linspace(-10,20,1000)
        # y=(1+  np.tanh((x - self.params_.blend_offset) * self.params_.blend_factor))*0.5
        # plt.plot(x,y)
        # plt.show()
        self.rhs = (1 - relation_dyn) * rhs_kin + relation_dyn * rhs_dyn
        self.f_ode = cs.Function('f_ode', [self.x_states, self.u_controls], [self.rhs])

    @property
    def n_x(self):
        return self.rhs.shape[0]
