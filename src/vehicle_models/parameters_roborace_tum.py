import matplotlib.pyplot as plt
import numpy as np
from model_dynamic import DynVehicleModelParameters, DriveParameters, TireParameters


def get_parameters() -> DynVehicleModelParameters:
    tire_parameters = TireParameters(
        B=10.0,
        C=2.5,
        Br=10.,
        Cr=2.5)
    drive_parameters = DriveParameters(
        c_air=0, #0.75
        c_rolling=0 #0.013
    )
    constraints = DynVehicleConstraints(
        maximum_acceleration_force=7e3,
        maximum_deceleration_force=20e3,
        maximum_velocity=60,  # Maximum lateral acceleration for both cars
        maximum_slip_angle=10 / 180 * np.pi,
        maximum_steering_rate=np.pi / 8,
        maximum_steering_angle=np.pi / 4 * 0.8
    )
    vehicle_parameters = DynVehicleModelParameters(
        length_front=1.6,
        length_rear=1.4,
        width=2,
        mass=1200,
        inertia=1200,
        street_mu=1,
        constraints=constraints,
        tires=tire_parameters,
        drive=drive_parameters
    )
    return vehicle_parameters


def plot_tire_curve(params: DynVehicleModelParameters):
    state_vx = 10
    state_vy = 0.0
    state_r = 0
    state_delta = np.linspace(0, np.pi / 16, 100)

    alpha_f = np.arctan((state_r * params.length_front + state_vy) / state_vx) - state_delta
    alpha_r = np.arctan((-state_r * params.length_rear + state_vy) / state_vx) + np.zeros_like(state_delta)

    # normal tire loads
    Fzf = params.mass * 9.81 * params.length_front / (params.length_front + params.length_rear)
    Fzr = params.mass * 9.81 * params.length_rear / (params.length_front + params.length_rear)

    # lateral tire forces
    Df = params.street_mu * Fzf
    Dr = params.street_mu * Fzr

    Ffy = Df * np.sin(params.tires.Cf * np.arctan(params.tires.Bf * alpha_f))
    Fry = Dr * np.sin(params.tires.Cr * np.arctan(params.tires.Br * alpha_r))

    plt.plot(alpha_f/np.pi*180, Ffy)
    plt.plot(alpha_f/np.pi*180, Fry)
    plt.show()
