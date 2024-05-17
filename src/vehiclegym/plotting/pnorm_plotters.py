from typing import TYPE_CHECKING, List
import matplotlib.cm as cm
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from vehicle_models.model_kinematic import KinematicModelParameters
from vehiclegym.animation.animation_helper import draw_vehicle

from vehiclegym.plotting.plotters import plot_road_f, plot_colourline_2, plot_road, plot_colourline_3
from vehiclegym.utils.automotive_datastructures import (CartesianTrajectory,
                                                        FrenetTrajectory)

if TYPE_CHECKING:
    from vehiclegym.road.road import Road
    from vehiclegym.planner.trajectory_planner_base import PlanningDataContainer
    from vehicle_models.model_kinematic import KinematicModelParameters


def plot_pnorm_trajectories(data_container: "PlanningDataContainer", road: "Road",
                            vehicle_parameter: List["KinematicModelParameters"], axs=None, plt_cbar=False):
    ego_traj = data_container[0].x
    t_traj = data_container[0].t
    nx, nstages, nsim = ego_traj.shape
    nopp = len(data_container) - 1
    if axs is None:
        fig, axs = plt.subplots(1, 1, figsize=(15, 3))
    axs.set_xlim((np.min(ego_traj[0, 0, :]) - 10, np.max(ego_traj[0, 0, :]) + 30))
    axs.set_ylim((-1.5, 1.5))

    # plot_road_f(road, fig=fig, axs=axs, fast_plot=True)
    x_min, x_max = np.min(ego_traj[0, ...]), np.max(ego_traj[0, ...])
    # axs.plot([x_min, x_max], [-road.nl_grid_[0]] * 2, color="grey")
    axs.plot([x_min - 100, x_max + 100], [0] * 2, "--", color="lightgrey")
    # axs.plot([x_min, x_max], [road.nr_grid_[0]] * 2, color="grey")

    for i_sim in range(5,nsim,5):
        plot_c_bar = True if i_sim == nsim - 1 and plt_cbar else False
        plot_colourline_2(axs, x=ego_traj[0, :, i_sim], y=ego_traj[1, :, i_sim], c=t_traj[:, i_sim] - t_traj[0, i_sim],
                          alpha=0.6,
                          plot_c_bar=plot_c_bar, label=r"prediction time $(s)$", c_ref=[0, t_traj[-1, 0]])
    axs.plot(ego_traj[0, 0, :], ego_traj[1, 0, :], color="black")

    if False:
        for i_snap in range(0, nsim, 3):
            ego_state = ego_traj[:, 0, i_snap]
            ego_state = FrenetTrajectory(ego_state)
            ego_state = road.transform_trajectory_f2c(ego_state)

            draw_vehicle(axs,
                         x_c=ego_state,
                         vehicle_length=vehicle_parameter[0].length,
                         chassis_length=vehicle_parameter[0].chassis_length,
                         chassis_width=vehicle_parameter[0].chassis_width,
                         tire_length=1,
                         tire_width=2.,linewidth=.5,
                         fast_plot=True, color="grey", alpha_shadow=1, fill=False)

    # draw vehicles
    for i_veh in range(1,nopp+1):
        opp_state = data_container[i_veh].x[:, 0, 0]
        opp_state = FrenetTrajectory(opp_state)
        opp_state = road.transform_trajectory_f2c(opp_state)

        draw_vehicle(axs,
                     x_c=opp_state,
                     vehicle_length=vehicle_parameter[i_veh].length,
                     chassis_length=vehicle_parameter[i_veh].chassis_length+vehicle_parameter[0].chassis_length+vehicle_parameter[0].safety_radius,
                     chassis_width=vehicle_parameter[i_veh].chassis_width+vehicle_parameter[0].chassis_width+vehicle_parameter[0].safety_radius,
                     tire_length=1,
                     tire_width=1,
                     linewidth=1,
                     fast_plot=True, color="tab:red", alpha_shadow=1, fill=True)

    if axs is None:
        plt.show()

def plot_pnrom_maneuver_trajectories(data_container_var1: "PlanningDataContainer",data_container_var2: "PlanningDataContainer", road: "Road",
                                    vehicle_parameter: List["KinematicModelParameters"], axs=None, plt_cbar=False,  xlim=None, ylim=None, type:int=1):
    ego_traj_var1 = data_container_var1[0].x
    ego_traj_var2 = data_container_var2[0].x
    t_traj = data_container_var1[0].t
    nx, nstages, nsim = ego_traj_var1.shape
    nopp = len(data_container_var1) - 1
    if axs is None:
        fig, axs = plt.subplots(1, 1, figsize=(15, 3))

    plot_road(road, fig=None, axs=axs)
    x_coord_min, x_coord_max, y_coord_min, y_coord_max = 1e6, -1e6, 1e6, -1e6

    if False:
        for i_sim in range(5,nsim):
            plot_c_bar = True if i_sim == nsim - 1 and plt_cbar else False

            # transform trajectoreis
            xf = FrenetTrajectory(ego_traj_var1[:, :, i_sim])
            xc = road.transform_trajectory_f2c(xf).get_as_array()
            x_coord_min = np.minimum(x_coord_min, np.min(xc[0, 0]))
            x_coord_max = np.maximum(x_coord_max, np.max(xc[0, 0]))
            y_coord_min = np.minimum(y_coord_min, np.min(xc[1, 0]))
            y_coord_max = np.maximum(y_coord_max, np.max(xc[1, 0]))
            plot_colourline_2(axs, x=xc[0, :], y=xc[1, :], c=t_traj[:, i_sim] - t_traj[0, i_sim],
                              alpha=0.1,
                              plot_c_bar=plot_c_bar, label=r"prediction time $(s)$", c_ref=[0, t_traj[-1, 0]])

            xf = FrenetTrajectory(ego_traj_var2[:, :, i_sim])
            xc = road.transform_trajectory_f2c(xf).get_as_array()
            x_coord_min = np.minimum(x_coord_min, np.min(xc[0, 0]))
            x_coord_max = np.maximum(x_coord_max, np.max(xc[0, 0]))
            y_coord_min = np.minimum(y_coord_min, np.min(xc[1, 0]))
            y_coord_max = np.maximum(y_coord_max, np.max(xc[1, 0]))
            plot_colourline_3(axs, x=xc[0, :], y=xc[1, :], c=t_traj[:, i_sim] - t_traj[0, i_sim],
                              alpha=0.1,
                              plot_c_bar=plot_c_bar, label=r"prediction time $(s)$", c_ref=[0, t_traj[-1, 0]])

    color_var1 = "tab:orange"
    color_var2 = "tab:blue"
    xf = FrenetTrajectory(ego_traj_var1[:, 0, :])
    xc = road.transform_trajectory_f2c(xf).get_as_array()
    axs.plot(xc[0, :], xc[1, :], color=color_var1,zorder=3)

    x_coord_min = np.minimum(x_coord_min, np.min(xc[0, :]))
    x_coord_max = np.maximum(x_coord_max, np.max(xc[0, :]))
    y_coord_min = np.minimum(y_coord_min, np.min(xc[1, :]))
    y_coord_max = np.maximum(y_coord_max, np.max(xc[1, :]))

    if type ==1:
        axs.scatter(xc[0, -1], xc[1, -1], color="black")
        axs.text(xc[0, -1], 2+xc[1, -1], r"$s_{\mathrm{max,}1}$")

    if type ==2:
        idx_min = np.argmin(-xf.get_as_array()[1,:])
        axs.scatter(xc[0, idx_min], xc[1, idx_min], color="black")
        axs.text(xc[0, idx_min], 2+xc[1, idx_min], r"$\Delta n_{\mathrm{max,}1}$")

    xf = FrenetTrajectory(ego_traj_var2[:, 0, :])
    xc = road.transform_trajectory_f2c(xf).get_as_array()
    axs.plot(xc[0, :], xc[1, :], color=color_var2, zorder=2)

    x_coord_min = np.minimum(x_coord_min, np.min(xc[0, :]))
    x_coord_max = np.maximum(x_coord_max, np.max(xc[0, :]))
    y_coord_min = np.minimum(y_coord_min, np.min(xc[1, :]))
    y_coord_max = np.maximum(y_coord_max, np.max(xc[1, :]))

    if type ==1:
        axs.scatter(xc[0, -1], xc[1, -1], color="black")
        axs.text(xc[0, -1], -8+xc[1, -1], r"$s_{\mathrm{max,}2}$")

    if type ==2:
        idx_min = np.argmin(-xf.get_as_array()[1,:])
        axs.scatter(xc[0, idx_min], xc[1, idx_min], color="black")
        axs.text(xc[0, idx_min], -5+xc[1, idx_min], r"$\Delta n_{\mathrm{max,}2}$")

    for i_snap in range(0, nsim, 5):
        ego_state = ego_traj_var1[:, 0, i_snap]
        ego_state = FrenetTrajectory(ego_state)
        ego_state = road.transform_trajectory_f2c(ego_state)

        draw_vehicle(axs,
                     x_c=ego_state,
                     vehicle_length=vehicle_parameter[0].length,
                     chassis_length=vehicle_parameter[0].chassis_length,
                     chassis_width=vehicle_parameter[0].chassis_width,
                     tire_length=1,
                     tire_width=0.2,
                     fast_plot=True, color=color_var1, alpha_shadow=0.2, fill=True, linewidth=1)

        ego_state = ego_traj_var2[:, 0, i_snap]
        ego_state = FrenetTrajectory(ego_state)
        ego_state = road.transform_trajectory_f2c(ego_state)

        draw_vehicle(axs,
                     x_c=ego_state,
                     vehicle_length=vehicle_parameter[0].length,
                     chassis_length=vehicle_parameter[0].chassis_length,
                     chassis_width=vehicle_parameter[0].chassis_width,
                     tire_length=1,
                     tire_width=0.2,
                     fast_plot=True, color=color_var2, alpha_shadow=0.2, fill=True, linewidth=1)

        # draw vehicles
        for i_veh in range(1,nopp+1):
            opp_state = data_container_var1[i_veh].x[:, 0, i_snap]
            opp_state = FrenetTrajectory(opp_state)
            opp_state = road.transform_trajectory_f2c(opp_state)

            draw_vehicle(axs,
                         x_c=opp_state,
                         vehicle_length=vehicle_parameter[i_veh].length,
                         chassis_length=vehicle_parameter[i_veh].chassis_length+vehicle_parameter[0].safety_radius,
                         chassis_width=vehicle_parameter[i_veh].chassis_width+vehicle_parameter[0].safety_radius,
                         tire_length=1,
                         tire_width=0.2,
                         linewidth=1,
                         fast_plot=True, color="tab:red", alpha_shadow=0.1, fill=True)

    if xlim is None:
        axs.set_xlim((x_coord_min-1, x_coord_max+1))
        axs.set_ylim((y_coord_min-1, y_coord_max+1))
    else:
        axs.set_xlim(xlim)
        axs.set_ylim(ylim)
    axs.grid('off')

    if axs is None:
        plt.show()


def plot_porm_driven(data_container: "PlanningDataContainer", road: "Road",
                     vehicle_parameter: List["KinematicModelParameters"], axs=None, color="black", linestyle="-", label=""):
    ego_traj = data_container[0].x
    t_traj = data_container[0].t
    nx, nstages, nsim = ego_traj.shape
    nopp = len(data_container) - 1
    if axs is None:
        fig, axs = plt.subplots(1, 1, figsize=(15, 3))
    axs.set_xlim((np.min(ego_traj[0, 0, :]) - 10, np.max(ego_traj[0, 0, :]) + 30))

    # plot_road_f(road, fig=fig, axs=axs, fast_plot=True)
    x_min, x_max = np.min(ego_traj[0, ...]), np.max(ego_traj[0, ...])
    # axs.plot([x_min, x_max], [-road.nl_grid_[0]] * 2, color="grey")
    axs.plot([x_min - 100, x_max + 100], [0] * 2, "--", color="grey")
    # axs.plot([x_min, x_max], [road.nr_grid_[0]] * 2, color="grey")

    axs.plot(ego_traj[0, 0, :], ego_traj[1, 0, :], color=color, linestyle=linestyle, label=label)

    if axs is None:
        plt.show()
