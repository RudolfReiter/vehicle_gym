import time
from copy import copy

import numpy as np
from matplotlib import pyplot as plt
from vehicle_models.model_kinematic import KinematicModelParameters

from vehiclegym.road.road import Road
from vehiclegym.planner.trajectory_planner_acados_20221 import PlannerOptions
from vehiclegym.utils.automotive_datastructures import (CartesianTrajectory,
                                                  FrenetTrajectory)
from vehicle_models.utils import get_circle_rect_cover, get_vertices_rectangle

SAVED_TIME = 0
INITIAL_TIME = 0

SAFETY_CIRCLE_COLORS = ""
ACCELERATION_ARROW_PLOTTED = False
from matplotlib.patches import Ellipse

def time_measure_start() -> None:
    global SAVED_TIME
    SAVED_TIME = time.time()


def time_measure_get_diff() -> float:
    global SAVED_TIME
    return time.time() - SAVED_TIME


def get_relative_time() -> float:
    global INITIAL_TIME
    if INITIAL_TIME < 1:
        INITIAL_TIME = time.time()
    return time.time() - INITIAL_TIME


def add_tire_rectangle(ax, xm: float, ym: float, alpha: float, delta: float, length: float, width: float,
                       alpha_shade=1):
    sum_angle = alpha + delta
    x0 = xm - length / 2 * np.cos(sum_angle)
    y0 = ym - length / 2 * np.sin(sum_angle)
    rectangle = plt.Rectangle((x0, y0),
                              length,
                              width,
                              angle=sum_angle / np.pi * 180,
                              fill=True,
                              color="black",
                              alpha=alpha_shade)
    ax.add_patch(rectangle)


def draw_vehicle_frenet(ax, road: Road, x_f: FrenetTrajectory, vehicle_length: float, chassis_length: float,
                        chassis_width: float, tire_length: float, tire_width: float, fast_plot: bool = False,
                        color="darkgrey", initial_guess_s: float = 0, alpha_shadow=1, simple_plot=True):
    if simple_plot:
        ax.plot(x_f.s, x_f.n, "o", color=color, linewidth=5, alpha=alpha_shadow)
    else:
        x_c = road.transform_trajectory_f2c(x_f)
        delta_rear = (chassis_length - vehicle_length) / 2

        x0 = x_c.x + vehicle_length / 2 * np.cos(x_c.phi)
        y0 = x_c.y + vehicle_length / 2 * np.sin(x_c.phi)
        n_total_per_edge = 5
        p_edges_c = get_vertices_rectangle(pos_center=(x0[0], y0[0]), dims=(chassis_length, chassis_width, x_c.phi[0]))
        p_edges_c_final = np.zeros((2, n_total_per_edge * 4 + 1))
        for i in range(n_total_per_edge):
            p_edges_c_final[:, i] = p_edges_c[:, 0] + i / n_total_per_edge * (p_edges_c[:, 1] - p_edges_c[:, 0])
            p_edges_c_final[:, i + n_total_per_edge] = p_edges_c[:, 1] + i / n_total_per_edge * (
                        p_edges_c[:, 2] - p_edges_c[:, 1])
            p_edges_c_final[:, i + 2 * n_total_per_edge] = p_edges_c[:, 2] + i / n_total_per_edge * (
                        p_edges_c[:, 3] - p_edges_c[:, 2])
            p_edges_c_final[:, i + 3 * n_total_per_edge] = p_edges_c[:, 3] + i / n_total_per_edge * (
                        p_edges_c[:, 0] - p_edges_c[:, 3])
        p_edges_c_final[:, -1] = p_edges_c_final[:, 0]
        p_edges_f_s = []
        p_edges_f_n = []
        for i in range(p_edges_c_final.shape[1]):
            edge_f = road.transform_trajectory_c2f(
                CartesianTrajectory(np.array([p_edges_c_final[0, i], p_edges_c_final[1, i], 0, 0, 0])),
                initial_guess_s=initial_guess_s)
            p_edges_f_s.append(edge_f.s)
            p_edges_f_n.append(edge_f.n)

        # p_edges_f_s.append(p_edges_f_s[0])
        # p_edges_f_n.append(p_edges_f_n[0])

        ax.plot(p_edges_f_s, p_edges_f_n, color=color, linewidth=1, alpha=alpha_shadow)


def draw_vehicle(ax, x_c: CartesianTrajectory, vehicle_length: float, chassis_length: float, chassis_width: float,
                 tire_length: float, tire_width: float, fast_plot: bool = False, color="darkgrey", alpha_shadow=1, fill: bool = True, linewidth:float=0.5):
    x_middle_rear_ax = x_c.x[0]
    y_middle_rear_ax = x_c.y[0]
    alpha = x_c.phi[0]
    delta = x_c.delta[0]

    # delta_rear
    delta_rear = (chassis_length - vehicle_length) / 2

    # Adding the 4 tires
    if not fast_plot:
        y0_rectangle_ego = y_middle_rear_ax + chassis_width / 2. * np.sin(alpha - np.pi / 2.)
        x0_rectangle_ego = x_middle_rear_ax + chassis_width / 2. * np.cos(alpha - np.pi / 2.)

        vec_len = chassis_width / 2.
        vec_angle = np.pi / 2.
        x_tire = x_middle_rear_ax + vec_len * np.cos(vec_angle + alpha)
        y_tire = y_middle_rear_ax + vec_len * np.sin(vec_angle + alpha)
        add_tire_rectangle(ax, x_tire, y_tire, alpha, 0., tire_length, tire_width, alpha_shade=alpha_shadow)

        vec_len = np.sqrt(chassis_width ** 2. + vehicle_length ** 2.)
        vec_angle = np.arctan(chassis_width / vehicle_length)
        x_tire = x0_rectangle_ego + vec_len * np.cos(vec_angle + alpha)
        y_tire = y0_rectangle_ego + vec_len * np.sin(vec_angle + alpha)
        add_tire_rectangle(ax, x_tire, y_tire, alpha, delta, tire_length, tire_width, alpha_shade=alpha_shadow)

        vec_len = vehicle_length
        vec_angle = 0.
        x_tire = x0_rectangle_ego + vec_len * np.cos(vec_angle + alpha)
        y_tire = y0_rectangle_ego + vec_len * np.sin(vec_angle + alpha)
        add_tire_rectangle(ax, x_tire, y_tire, alpha, np.pi + delta, tire_length, tire_width, alpha_shade=alpha_shadow)

        x_tire = x0_rectangle_ego
        y_tire = y0_rectangle_ego
        add_tire_rectangle(ax, x_tire, y_tire, alpha, np.pi, tire_length, tire_width, alpha_shade=alpha_shadow)

    # Compute vehicle center position
    y0_rectangle_ego = y_middle_rear_ax + chassis_width / 2. * np.sin(alpha - np.pi / 2.) - delta_rear * np.sin(alpha)
    x0_rectangle_ego = x_middle_rear_ax + chassis_width / 2. * np.cos(alpha - np.pi / 2.) - delta_rear * np.cos(alpha)
    rectangle = plt.Rectangle((x0_rectangle_ego, y0_rectangle_ego),
                              chassis_length,
                              chassis_width,
                              angle=alpha / np.pi * 180.,
                              fill=fill,
                              facecolor=color,
                              edgecolor="grey",
                              linewidth=linewidth,
                              alpha=alpha_shadow)
    #ell = Ellipse(xy=(x0_rectangle_ego+chassis_length/2, y0_rectangle_ego+chassis_width/2), width=chassis_length*np.sqrt(2), height=chassis_width*np.sqrt(2), fc="lightgrey",ec="tab:orange")
    #ax.add_patch(ell)

    if fill:
        ax.add_patch(rectangle)
        rectangle = plt.Rectangle((x0_rectangle_ego, y0_rectangle_ego),
                                  chassis_length,
                                  chassis_width,
                                  angle=alpha / np.pi * 180.,
                                  fill=False,
                                  facecolor=color,
                                  edgecolor="grey",
                                  linewidth=linewidth,
                                  alpha=1)
    ax.add_patch(rectangle)


def plot_arrow_vehicle_c(ax, x_c: CartesianTrajectory, length: float, angle_arrow: float,
                         color: str = "black", max_value=1., min_value=1., scaling: float = 1):
    total_angle = x_c.phi[0] + angle_arrow

    dx_arrow = max_value * np.cos(total_angle) / max_value * scaling
    dy_arrow = max_value * np.sin(total_angle) / max_value * scaling
    arrow = plt.Arrow(x_c.x[0], x_c.y[0], dx_arrow, dy_arrow, color=color, alpha=0.1)
    ax.add_patch(arrow)

    dx_arrow = min_value * np.cos(total_angle) / max_value * scaling
    dy_arrow = min_value * np.sin(total_angle) / max_value * scaling
    arrow = plt.Arrow(x_c.x[0], x_c.y[0], dx_arrow, dy_arrow, color=color, alpha=0.1)
    ax.add_patch(arrow)

    dx_arrow = length * np.cos(total_angle) / max_value * scaling
    dy_arrow = length * np.sin(total_angle) / max_value * scaling
    arrow = plt.Arrow(x_c.x[0], x_c.y[0], dx_arrow, dy_arrow, color=color)
    ax.add_patch(arrow)


def plot_arrow_f2c(ax, road, s0: float, n0: float, ds: float, dn: float, color: str = "black",
                   max_value=None, alpha: float = 1.):
    xf0 = FrenetTrajectory()
    xf0.s = s0
    xf0.n = n0
    xc0 = road.transform_trajectory_f2c(xf0)

    xf_arrow_hat = FrenetTrajectory()
    xf_arrow_hat.s = s0 + ds
    xf_arrow_hat.n = n0 + dn
    xc_arrow_hat = road.transform_trajectory_f2c(xf_arrow_hat)

    dx_arrow = xc_arrow_hat.x - xc0.x
    dy_arrow = xc_arrow_hat.y - xc0.y

    arrow = plt.Arrow(xc0.x, xc0.y, dx_arrow, dy_arrow, color=color, alpha=alpha)
    ax.add_patch(arrow)

    if max_value is not None:
        circle1 = plt.Circle((xc0.x, xc0.y),
                             max_value,
                             fill=False,
                             linewidth=0.5,
                             alpha=0.5,
                             color=color)
        ax.add_patch(circle1)


def plot_covering_safety_circles(ax, road: Road,
                                 planner_options: PlannerOptions,
                                 vehicle_state_f: FrenetTrajectory,
                                 vehicle_parameters: KinematicModelParameters,
                                 color="black",
                                 alpha=1):
    xf_ego = FrenetTrajectory(np.array([vehicle_state_f.s, vehicle_state_f.n, vehicle_state_f.alpha, 0, 0]))
    xc_ego = road.transform_trajectory_f2c(xf_ego)
    positions_x, positions_y, radius_auto = get_circle_rect_cover(n_circ=planner_options.n_circles_ego,
                                                                  wheelbase=vehicle_parameters.length,
                                                                  l_rect=vehicle_parameters.chassis_length,
                                                                  w_rect=vehicle_parameters.chassis_width,
                                                                  state_s=xc_ego.x,
                                                                  state_n=xc_ego.y,
                                                                  state_alpha=xc_ego.phi)
    for pos_x, pos_y in zip(positions_x, positions_y):
        if planner_options.auto_size_circles:
            radius = radius_auto
        else:
            radius = vehicle_parameters.safety_radius

        plot_safety_circles(ax, pos_x, pos_y, radius, None, color=color, alpha_nom=alpha)


def plot_safety_circles(ax, x0, y0, safety_radius_strict, safety_radius_minor=None, color="black", alpha_nom=1,
                        zorder=10):
    global SAFETY_CIRCLE_COLORS
    if safety_radius_minor is not None:
        circle1 = plt.Circle((x0, y0),
                             safety_radius_minor,
                             fill=True,
                             color=color,
                             alpha=alpha_nom / 2,
                             linewidth=0.1,
                             zorder=zorder)
        ax.add_patch(circle1)

    if color == SAFETY_CIRCLE_COLORS:
        label_c = ""
    else:
        SAFETY_CIRCLE_COLORS = color
        label_c = "safety circle"
    circle1 = plt.Circle((x0, y0),
                         safety_radius_strict,
                         fill=True,
                         color=color,
                         alpha=alpha_nom,
                         zorder=zorder + 1,
                         label=label_c)
    ax.add_patch(circle1)
