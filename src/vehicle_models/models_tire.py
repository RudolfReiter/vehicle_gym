from abc import ABC, abstractmethod
from dataclasses import dataclass
import casadi as cs
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from vehicle_models.model_dynamic import DynVehicleModelParameters


@dataclass
class TireParameters(ABC):
    pass


@dataclass
class LinearTireParameters(TireParameters):
    c_lin: float = 78480  # Computed out of pacejka parameters below


@dataclass
class PacejkaTireParameters(TireParameters):
    B: float = 10.0
    C: float = 2.5
    mu: float = 1


class TireModel(ABC):
    def __init__(self, params):
        self.sum_pressure_rear = None
        self.sum_pressure_front = None
        self.force_long_front_single_tire = None
        self.force_long_rear_single_tire = None
        self.params = params
        self.state_vx = None
        self.state_vy = None
        self.state_omega = None

        self.control_force_tires = None
        self.control_delta = None

        self.control_torque_rear = None
        self.control_brake_press_rear = None
        self.control_brake_press_front = None

    def set_variables(self, x_states, u_controls):
        self.state_vx = x_states[3]
        self.state_vy = x_states[4]
        self.state_omega = x_states[5]

        self.control_force_tires = u_controls[0]  # Force on tires
        self.control_delta = u_controls[1]
        self.control_torque_rear = u_controls[2]  # Torque acts on both wheels
        self.control_brake_press_front = u_controls[3]  # brake pressure acts on one wheel only
        self.control_brake_press_rear = u_controls[4]

        acc_force = cs.fmax(0, self.control_force_tires)
        dec_force = cs.fmax(0, -self.control_force_tires)

        # Brake pressure needs the right sign
        self.force_long_rear_single_tire = 1 / 2 * (self.control_torque_rear / self.params.wheel_radius_rear) - \
                                           cs.tanh(
                                               1e3 * self.state_vx) * self.control_brake_press_rear * self.params.drive.factor_pressure2force_rear + \
                                           1 / 2 * acc_force - \
                                           1 / 2 * self.params.drive.k_brake_r * dec_force

        self.force_long_front_single_tire = cs.tanh(1e3 * self.state_vx) * (-self.control_brake_press_front) * \
                                            self.params.drive.factor_pressure2force_front - \
                                            1 / 2 * (1 - self.params.drive.k_brake_r) * dec_force

        self.sum_pressure_front = self.control_brake_press_front + 1 / 2 * (1 - self.params.drive.k_brake_r) \
                                  * dec_force * 1 / self.params.drive.factor_pressure2force_front * cs.tanh(
            1e3 * self.state_vx)

        self.sum_pressure_rear = self.control_brake_press_rear + 1 / 2 * (self.params.drive.k_brake_r) \
                                 * dec_force * 1 / self.params.drive.factor_pressure2force_front * cs.tanh(
            1e3 * self.state_vx)

    # Input is counted totally. Needs translation 2 single tire

    @abstractmethod
    def get_forces(self):
        pass


class LinearTireModel(TireModel):
    def __int__(self, states, controls, params: "DynVehicleModelParameters"):
        super().__init__(params)

    def get_forces(self):
        # longitudinal tire forces
        Frl = self.force_long_rear_single_tire
        Ffl = self.force_long_front_single_tire

        # slip parameters
        alpha_f = self.control_delta - cs.arctan2((self.state_omega * self.params.length_front + self.state_vy),
                                                  self.state_vx)
        alpha_r = -cs.arctan2((-self.state_omega * self.params.length_rear + self.state_vy), self.state_vx)

        # lateral tire forces
        Ffc = alpha_f * self.params.tires.c_lin
        Frc = alpha_r * self.params.tires.c_lin

        # vehicle frame forces
        Ffx = Ffl * cs.cos(self.control_delta) - Ffc * cs.sin(self.control_delta)
        Frx = Frl
        Ffy = Ffl * cs.sin(self.control_delta) + Ffc * cs.cos(self.control_delta)
        Fry = Frc

        return Ffx, Frx, Ffy, Fry

    def get_tire_force_function(self):
        """
        Function for plotting and comparing the force
        :return: function with Force = f(alpha)
        """
        force_func = lambda alpha: alpha * self.params.tires.c_lin
        return force_func


class PacejkaTireModel(TireModel):
    def __int__(self, states, controls, params: "DynVehicleModelParameters"):
        super().__init__(params)

    def get_forces(self):
        Frl = self.force_long_rear_single_tire
        Ffl = self.force_long_front_single_tire

        # slip parameters
        alpha_f = self.control_delta - cs.arctan2((self.state_omega * self.params.length_front + self.state_vy),
                                                  self.state_vx)
        alpha_r = -cs.arctan2((-self.state_omega * self.params.length_rear + self.state_vy), self.state_vx)

        # lateral tire forces
        # normal tire loads
        Fzf = 1 / 2 * self.params.mass * 9.81 * self.params.length_front / \
              (self.params.length_front + self.params.length_rear)
        Fzr = 1 / 2 * self.params.mass * 9.81 * self.params.length_rear / \
              (self.params.length_front + self.params.length_rear)

        Df = self.params.tires.mu * Fzf
        Dr = self.params.tires.mu * Fzr

        Ffc = Df * cs.sin(self.params.tires.C * cs.arctan2(self.params.tires.B * alpha_f, 1))
        Frc = Dr * cs.sin(self.params.tires.C * cs.arctan2(self.params.tires.B * alpha_r, 1))

        # vehicle frame forces
        Ffx = Ffl * cs.cos(self.control_delta) - Ffc * cs.sin(self.control_delta)
        Frx = Frl
        Ffy = Ffl * cs.sin(self.control_delta) + Ffc * cs.cos(self.control_delta)
        Fry = Frc

        return Ffx, Frx, Ffy, Fry

    def get_tire_force_function(self):
        """
        Function for plotting and comparing the force
        :return: function with Force = f(alpha)
        """
        Fzf = 1 / 2 * self.params.mass * 9.81 * self.params.length_front / \
              (self.params.length_front + self.params.length_rear)
        Df = self.params.tires.mu * Fzf
        force_func = lambda alpha: Df * np.sin(self.params.tires.C * cs.arctan2(self.params.tires.B * alpha, 1))
        return force_func

    def get_lin_coef(self):
        """
        Function that returns stiffness at alpha=0
        :return: c_lin
        """
        force_fun = self.get_tire_force_function()
        alpha_0 = 0
        alpha_1 = 1e-5
        c_lin = (force_fun(alpha_1) - force_fun(alpha_0)) / (alpha_1 - alpha_0)
        return c_lin
