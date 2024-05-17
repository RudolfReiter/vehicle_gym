import pickle
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
import io
from enum import Enum, auto
from typing import TYPE_CHECKING, List, Tuple
import matplotlib as mpl
import numpy as np
import seaborn as sns
from PIL import Image

from matplotlib.animation import FuncAnimation, FFMpegWriter
from vehiclegym.simulator.simulator_simple import SimulationStatistics
from vehiclegym.utils.helpers import FrozenClass, latexify
from vehiclegym.animation.animation_helper import *
from vehiclegym.plotting.plotters import plot_road, plot_road_f
from vehiclegym.road.road import Road
from vehiclegym.simulator.simulation_datastructures import RacingSimulationData
from vehiclegym.planner.trajectory_planner_base import PlanningDataContainer

if TYPE_CHECKING:
    from vehiclegym.planner.trajectory_planner_acados_20221 import PlanningDataContainer, PlannerOptionsAcados
    from vehicle_models.model_kinematic import KinematicModelParameters


class AnimationPlanningColorType(Enum):
    UNI = auto()
    VELOCITY = auto()
    TIME = auto()
    TEAMS = auto()
    BLACK = auto()


class AnimationPanels(Enum):
    CARTESIAN = auto()
    FRENET = auto()
    CARTESIAN_FRENET = auto()


PREDICTION_PLOTTED = False
PLAN_PLOTTED = []


@dataclass
class AnimationParameters(FrozenClass):
    """ Parameters for kinematic vehicle models"""
    animation_frames_per_sec: int = 10  # Hz Can't be higher than maximum drawing rate (around 70ms)
    animation_speed: float = 1
    animation_tire_length: float = 0.5
    animation_tire_width: float = 0.15
    figure_additional_size: float = 50
    fast_animation: bool = True
    color_palette: str = 'inferno_r'  # "hls", 'cividis', cividis_r, copper_r, inferno_r
    planning_color_type: AnimationPlanningColorType = AnimationPlanningColorType.UNI
    index_of_simulated_steps: int = 1  # The mpc trajectory is controlled until this index, before the rl agent takes
    plot_opp_predictions: List[int] = field(default_factory=lambda: [])
    plot_ego_plans: List[int] = field(default_factory=lambda: [])
    plot_acceleration_arrows: List[int] = field(default_factory=lambda: [])
    plot_safety_circles: List[int] = field(default_factory=lambda: [])
    plot_team_idxs: List[int] = field(default_factory=lambda: [])
    supress_warnings: bool = True
    animation_pannels: AnimationPanels = AnimationPanels.CARTESIAN
    animation_focus_ego: bool = True
    repeat: bool = True
    show_title_str: bool = True


class Animator:
    def __init__(self, animation_parameter: AnimationParameters):
        self.parameter = animation_parameter
        self.planning_data_container = None
        self.vehicle_parameter: KinematicModelParameters = None
        self.planner_parameter: PlannerOptionsAcados = None
        self.statistics = None
        self.fig = None
        self.ax = None
        self.road = None
        self.n_vehicles = None
        self.colors = None
        self.frenet_road = None
        self.static_obstacle_states = None
        self.static_obstacle_params = None
        self.x_coord_min, self.x_coord_max, self.y_coord_min, self.y_coord_max = None, None, None, None

    def set_data(self,
                 planning_data_container: List["PlanningDataContainer"],
                 vehicle_parameter: List["KinematicModelParameters"],
                 planner_parameter: List["PlannerOptionsAcados"],
                 road: Road,
                 static_obstacle_params: List[np.ndarray] = None,
                 static_obstacle_states: List[np.ndarray] = None,
                 statistics: SimulationStatistics = None):

        # Check lengths of arrays and safe to internal sturctures
        self.n_vehicles = len(planning_data_container)
        assert len(planner_parameter) == self.n_vehicles
        assert len(vehicle_parameter) == self.n_vehicles
        assert len(self.parameter.plot_safety_circles) <= self.n_vehicles
        assert len(self.parameter.plot_acceleration_arrows) <= self.n_vehicles
        assert len(self.parameter.plot_ego_plans) <= self.n_vehicles
        assert len(self.parameter.plot_opp_predictions) <= self.n_vehicles

        self.static_obstacle_states = static_obstacle_states
        self.static_obstacle_params = static_obstacle_params

        # Hackmack to imitate parameters for data that do not come from acados planner
        fake_planner_parameter = PlannerOptions()
        acados_planners_times = [pp.time_disc for pp in planner_parameter if pp is not None]
        if len(acados_planners_times) > 0:
            fake_planner_parameter.time_disc = acados_planners_times[0]
        else:
            fake_planner_parameter.time_disc = planning_data_container[0].t[1] - planning_data_container[0].t[0]
        fake_planner_parameter.n_nodes = 1
        planner_parameter = [pp if pp is not None else fake_planner_parameter for pp in planner_parameter]

        # Hackmack to transform frenet trajectories into planning data containers
        for i, container in enumerate(planning_data_container):
            if isinstance(container, PlanningDataContainer):
                planning_data_container[i].flatten_data()
            elif isinstance(container, FrenetTrajectory):
                container_new = PlanningDataContainer()
                container_new.x = container.get_as_array()
                container_new.x_flat = container.get_as_array()
                container_new.t = container.t
                container_new.t_flat = container.t
                planning_data_container[i] = container_new
            else:
                Exception()

        self.planning_data_container = planning_data_container
        self.vehicle_parameter = vehicle_parameter
        self.planner_parameter = planner_parameter
        self.road = road
        self.frenet_road = deepcopy(road)
        self.frenet_road.set_kappa(np.zeros_like(self.frenet_road.kappa_grid_) + 1e-6)
        self.statistics = statistics

        # resample data for animation
        resample_delta_t = 1 / self.parameter.animation_frames_per_sec * self.parameter.animation_speed
        for i in range(self.n_vehicles):
            self.planning_data_container[i].resample(self.planner_parameter[i].time_disc, resample_delta_t)

        # get colors
        # sns.reset_orig()  # get default matplotlib styles back
        self.colors = sns.color_palette(self.parameter.color_palette, n_colors=self.n_vehicles)
        self.velocity_cmap = plt.get_cmap('winter')
        self.color_speed_max = 0
        for i in range(self.n_vehicles):
            self.color_speed_max = np.maximum([vehicle_parameter[i].maximum_velocity], self.color_speed_max)

    def plot_range_cartesian(self,  i_range, filename: str, size: Tuple[int, int],title_str:str = ''):
        latexify(fontsize=15)
        self.fig, ax = plt.subplots(nrows=1, ncols=1, figsize=size)
        self.ax = ax
        for i in i_range:
            self.animate_cartesian(i, clear=False, static_colors=("springgreen", "white","white"))

        if self.parameter.planning_color_type == AnimationPlanningColorType.VELOCITY:
            v_max = np.max([vehicle_p.maximum_velocity for vehicle_p in self.vehicle_parameter])
            norm = mpl.colors.Normalize(vmin=0, vmax=v_max)
            sm = plt.cm.ScalarMappable(cmap=self.velocity_cmap, norm=norm)
            sm.set_array([])
            self.fig.colorbar(sm, ax=self.ax, label=r"v ($\frac{m}{s}$)")
        elif self.parameter.planning_color_type == AnimationPlanningColorType.TIME:
            t_max = self.planner_parameter[0].n_nodes * self.planner_parameter[0].time_disc
            norm = mpl.colors.Normalize(vmin=0, vmax=t_max)
            sm = plt.cm.ScalarMappable(cmap=self.velocity_cmap, norm=norm)
            sm.set_array([])
            self.fig.colorbar(sm, ax=self.frenet_ax, label=r"MPC trajectory $x(t)$ in ($s$)")

        ax.set_title(title_str)
        plt.tight_layout()
        plt.savefig(filename + ".pdf", dpi=300)

    def plot_frame(self, i, filename: str, size: Tuple[int, int]):
        latexify(fontsize=10)
        self.fig, axs = plt.subplots(nrows=1, ncols=2, figsize=size)
        self.ax = axs[0]
        self.frenet_ax = axs[1]
        self.animate_two_frames(i)

        if self.parameter.planning_color_type == AnimationPlanningColorType.VELOCITY:
            v_max = np.max([vehicle_p.maximum_velocity for vehicle_p in self.vehicle_parameter])
            norm = mpl.colors.Normalize(vmin=0, vmax=v_max)
            sm = plt.cm.ScalarMappable(cmap=self.velocity_cmap, norm=norm)
            sm.set_array([])
            self.fig.colorbar(sm, ax=self.ax, label=r"v ($\frac{m}{s}$)")
        elif self.parameter.planning_color_type == AnimationPlanningColorType.TIME:
            t_max = self.planner_parameter[0].n_nodes * self.planner_parameter[0].time_disc
            norm = mpl.colors.Normalize(vmin=0, vmax=t_max)
            sm = plt.cm.ScalarMappable(cmap=self.velocity_cmap, norm=norm)
            sm.set_array([])
            self.fig.colorbar(sm, ax=self.frenet_ax, label=r"MPC trajectory $x(t)$ in ($s$)")

        plt.tight_layout()
        plt.savefig(filename + ".pdf", dpi=300)

    def animate(self, save_as_movie: bool = False, f_name: str = r"animation.mp4",
                fig_size=(25,15), title_str: str = None, use_latex:bool=False, static_colors=tuple()):
        use_latex and latexify(fontsize=18)
        if self.parameter.animation_pannels == AnimationPanels.CARTESIAN:
            self.fig, self.ax = plt.subplots(figsize=fig_size)
            ani_func = self.animate_cartesian
        elif self.parameter.animation_pannels == AnimationPanels.FRENET:
            self.fig, self.frenet_ax = plt.subplots(figsize=fig_size)
            ani_func = self.animate_frenet
        elif self.parameter.animation_pannels == AnimationPanels.CARTESIAN_FRENET:
            self.fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(40, 20))
            self.ax = axs[0]
            self.frenet_ax = axs[1]
            ani_func = self.animate_two_frames
        else:
            ani_func = None
            Exception()

        if self.parameter.planning_color_type == AnimationPlanningColorType.VELOCITY:
            v_max = np.max([vehicle_p.maximum_velocity for vehicle_p in self.vehicle_parameter])
            norm = mpl.colors.Normalize(vmin=0, vmax=v_max)
            sm = plt.cm.ScalarMappable(cmap=self.velocity_cmap, norm=norm)
            sm.set_array([])
            self.fig.colorbar(sm, ax=self.ax, label=r"v ($\frac{m}{s}$)")
        elif self.parameter.planning_color_type == AnimationPlanningColorType.TIME:
            t_max = self.planner_parameter[0].n_nodes * self.planner_parameter[0].time_disc
            norm = mpl.colors.Normalize(vmin=0, vmax=t_max)
            sm = plt.cm.ScalarMappable(cmap=self.velocity_cmap, norm=norm)
            sm.set_array([])
            self.fig.colorbar(sm, ax=self.ax, label=r"t ($s$)")
        (_, n_samples) = self.planning_data_container[0].x_flat.shape

        ani = FuncAnimation(self.fig, ani_func,
                            fargs=(True, static_colors, title_str),
                            interval=1,
                            frames=n_samples,
                            repeat=self.parameter.repeat,
                            repeat_delay=1000)
        if save_as_movie:
            writervideo = FFMpegWriter(fps=self.parameter.animation_frames_per_sec)
            ani.save(f_name, writer=writervideo)
        else:
            plt.show()

    def animate_two_frames(self, i):
        self.animate_cartesian(i)
        self.animate_frenet(i)

    def animate_cartesian(self, i, clear=True, static_colors=tuple(), title_str_input: str = None):
        global PREDICTION_PLOTTED, PLAN_PLOTTED
        PREDICTION_PLOTTED = False
        PLAN_PLOTTED = []
        # Measure execution time and relate it to animation speed
        time_from_lat_call = time_measure_get_diff()
        if not self.parameter.supress_warnings:
            if time_from_lat_call > 1 / self.parameter.animation_frames_per_sec:
                print("Animation plot is too slow ({:4.3f}s) for the requested frames per second ({}f/s)!".format(
                    time_from_lat_call,
                    self.parameter.animation_frames_per_sec))

        time.sleep(np.maximum(0, 1 / self.parameter.animation_frames_per_sec - time_from_lat_call))
        time_measure_start()

        # Plot road
        if clear:
            self.ax.clear()

        self.ax.axis('equal')
        plot_road(self.road, self.fig, self.ax, fast_plot=self.parameter.fast_animation)

        # plot static obstacles
        if self.static_obstacle_states is not None:
            for static_obstacle, params in zip(self.static_obstacle_states, self.static_obstacle_params):
                xf = FrenetTrajectory(static_obstacle)
                xc = self.road.transform_trajectory_f2c(xf).get_as_array()
                ellipse = Ellipse(xy=(xc[0], xc[1]),
                                  width=params.chassis_length * np.sqrt(2) + params.safety_radius_length,
                                  height=params.chassis_width * np.sqrt(2) + params.safety_radius_boundary,
                                  angle=np.rad2deg(xc[2][0]),
                                  edgecolor='black', fc='tab:red', lw=1)
                self.ax.add_patch(ellipse)

        # create string for animation information
        title_str = ""
        t_current = 0
        x_coord_min, x_coord_max, y_coord_min, y_coord_max = 1e6, -1e6, 1e6, -1e6

        # iterate through vehicles
        for i_vehicle in range(self.n_vehicles):
            # Get data
            t_current = self.planning_data_container[i_vehicle].t_flat[i]
            xf_ego = FrenetTrajectory(self.planning_data_container[i_vehicle].x_flat[:, i])
            xc_ego = self.road.transform_trajectory_f2c(xf_ego)
            x_ego = xc_ego.get_as_array()
            a_lat_ego = -x_ego[3] * x_ego[3] / self.vehicle_parameter[i_vehicle].length * np.tan(x_ego[4])
            alpha_ego = x_ego[2]
            x_center_ego = x_ego[0] + self.vehicle_parameter[i_vehicle].length / 2 * np.cos(alpha_ego)
            y_center_ego = x_ego[1] + self.vehicle_parameter[i_vehicle].length / 2 * np.sin(alpha_ego)

            title_str += r"v{}={:4.1f} ".format(i_vehicle, float(x_ego[3])) + r"$\frac{m}{s}$, "

            if not (self.parameter.animation_focus_ego and i_vehicle > 0):
                x_coord_min = np.minimum(x_coord_min, x_ego[0])
                x_coord_max = np.maximum(x_coord_max, x_ego[0])
                y_coord_min = np.minimum(y_coord_min, x_ego[1])
                y_coord_max = np.maximum(y_coord_max, x_ego[1])

            # Draw safety radius
            if self.planner_parameter[i_vehicle] is not None and self.planner_parameter[i_vehicle].use_lifting:
                if i_vehicle in self.parameter.plot_safety_circles:
                    plot_safety_circles(self.ax, x_center_ego, y_center_ego,
                                        self.vehicle_parameter[i_vehicle].safety_radius,
                                        None,
                                        color=self.colors[i_vehicle], alpha_nom=0.2)

                    plot_covering_safety_circles(self.ax, self.road,
                                                 self.planner_parameter[i_vehicle],
                                                 xf_ego,
                                                 self.vehicle_parameter[i_vehicle],
                                                 color=self.colors[i_vehicle],
                                                 alpha=0.7)

            # Fetch color
            color = "darkgrey" if not static_colors else static_colors[i_vehicle]

            # Draw vehicle
            draw_vehicle(self.ax, xc_ego,
                         self.vehicle_parameter[i_vehicle].length,
                         self.vehicle_parameter[i_vehicle].chassis_length,
                         self.vehicle_parameter[i_vehicle].chassis_width,
                         self.parameter.animation_tire_length,
                         self.parameter.animation_tire_width,
                         fast_plot=self.parameter.fast_animation,
                         color=color)

            # Acceleration Arrows
            if self.planning_data_container[i_vehicle].u_flat.shape[0] > 0 and \
                    i_vehicle in self.parameter.plot_acceleration_arrows:
                u_ego = self.planning_data_container[i_vehicle].u_flat[:, i]
                # longitudinal acceleration arrow
                plot_arrow_vehicle_c(ax=self.ax, x_c=xc_ego, length=u_ego[0], angle_arrow=0, color="red",
                                     max_value=self.vehicle_parameter[i_vehicle].maximum_acceleration_force,
                                     min_value=-self.vehicle_parameter[i_vehicle].maximum_deceleration_force,
                                     scaling=4)

            # lateral acceleration arrow
            if i_vehicle in self.parameter.plot_acceleration_arrows:
                plot_arrow_vehicle_c(ax=self.ax, x_c=xc_ego, length=a_lat_ego, angle_arrow=np.pi / 2, color="red",
                                     max_value=self.vehicle_parameter[i_vehicle].maximum_lateral_acc,
                                     min_value=-self.vehicle_parameter[i_vehicle].maximum_lateral_acc,
                                     scaling=4)

            # Plot predictions
            if i_vehicle in self.parameter.plot_opp_predictions:
                for i_opponent in range(self.n_vehicles - 1):
                    if not self.planning_data_container[i_vehicle].x_opp_predicted_all:
                        #print("No predictions available for vehicle {}!".format(i_vehicle))
                        pass
                    else:
                        traj_idx = int(
                            np.floor(i / self.planning_data_container[i_vehicle].n_trajectory_states_truncated))
                        x_pred = self.planning_data_container[i_vehicle].x_opp_predicted_all[i_opponent][:, :, traj_idx]
                        p_x = []
                        p_y = []
                        for index in range(0, x_pred.shape[1]):
                            xf = FrenetTrajectory(x_pred[:, index])
                            x = self.road.transform_trajectory_f2c(xf).get_as_array()
                            x_center = x[0] + self.vehicle_parameter[1].length / 2 * np.cos(x[2])
                            y_center = x[1] + self.vehicle_parameter[1].length / 2 * np.sin(x[2])
                            p_x.append(x_center[0])
                            p_y.append(y_center[0])
                        label = "prediction" if not PREDICTION_PLOTTED else ""
                        PREDICTION_PLOTTED = True
                        self.ax.plot(p_x, p_y, alpha=0.6, zorder=-2, color="lightgrey", linewidth=10, label=label)

            # Plot plans
            if i_vehicle in self.parameter.plot_ego_plans:
                if self.planning_data_container[i_vehicle].n_trajectory_states_truncated > 0:
                    traj_idx = int(np.floor(i / self.planning_data_container[i_vehicle].n_trajectory_states_truncated))
                    x_pred = self.planning_data_container[i_vehicle].x[:, :, traj_idx]
                    p_x = []
                    p_y = []
                    colors = []
                    for index in range(0, x_pred.shape[1]):
                        # Transform to cartesian
                        xf = FrenetTrajectory(x_pred[:, index])
                        x = self.road.transform_trajectory_f2c(xf).get_as_array()
                        x_center = x[0] + self.vehicle_parameter[i_vehicle].length / 2 * np.cos(x[2])
                        y_center = x[1] + self.vehicle_parameter[i_vehicle].length / 2 * np.sin(x[2])

                        # Different color codings for ego plan
                        p_x.append(x_center[0])
                        p_y.append(y_center[0])
                        if self.parameter.planning_color_type == AnimationPlanningColorType.VELOCITY:
                            v_max = np.max([vehicle_p.maximum_velocity for vehicle_p in self.vehicle_parameter])
                            color_iter = (x[3] / v_max)[0]
                        elif self.parameter.planning_color_type == AnimationPlanningColorType.TIME:
                            color_iter = index / x_pred.shape[1]
                        else:
                            color_iter = self.colors[i_vehicle]
                        colors.append(color_iter)
                    if self.parameter.planning_color_type == AnimationPlanningColorType.UNI:
                        label = "planned trajectory" if not (i_vehicle in PLAN_PLOTTED) else ""
                        PLAN_PLOTTED.append(i_vehicle)
                        self.ax.plot(p_x, p_y, alpha=0.6, zorder=-2, color=self.colors[i_vehicle], linewidth=3,
                                     label=label)
                    elif self.parameter.planning_color_type == AnimationPlanningColorType.TEAMS:
                        if i_vehicle in self.parameter.plot_team_idxs:
                            color = "limegreen"
                        else:
                            color = "red"
                        label = "planned trajectory" if not (i_vehicle in PLAN_PLOTTED) else ""
                        PLAN_PLOTTED.append(i_vehicle)
                        self.ax.plot(p_x, p_y, alpha=0.6, zorder=-2, color=color, linewidth=10, label=label)

                    else:
                        self.ax.scatter(p_x, p_y, alpha=0.6, zorder=-2, s=100, c=colors, cmap=self.velocity_cmap)

            if self.statistics is not None:
                idx_statistics = int(np.floor(t_current / self.statistics.delta_t))
                if self.statistics.collisions_follow[i_vehicle, idx_statistics] > 0:
                    draw_vehicle(self.ax, xc_ego,
                                 self.vehicle_parameter[i_vehicle].length,
                                 self.vehicle_parameter[i_vehicle].chassis_length,
                                 self.vehicle_parameter[i_vehicle].chassis_width,
                                 self.parameter.animation_tire_length,
                                 self.parameter.animation_tire_width,
                                 fast_plot=True,
                                 color="red")
                if self.statistics.collisions_lead[i_vehicle, idx_statistics] > 0:
                    draw_vehicle(self.ax, xc_ego,
                                 self.vehicle_parameter[i_vehicle].length,
                                 self.vehicle_parameter[i_vehicle].chassis_length,
                                 self.vehicle_parameter[i_vehicle].chassis_width,
                                 self.parameter.animation_tire_length,
                                 self.parameter.animation_tire_width,
                                 fast_plot=True,
                                 color="green")

        # set title and zoom
        title_str = r"$t_{real}$ = " + "{:.2f}s, ".format(get_relative_time()) + \
                    r"$t_{sim}$ = " + "{:.2f}s, ".format(t_current) + title_str
        if self.parameter.show_title_str:
            if self.parameter.animation_pannels == AnimationPanels.CARTESIAN:
                self.ax.set_title(title_str)
            else:
                self.ax.set_title("Cartesian Coordinate Frame")
        if title_str_input is not None:
            self.ax.set_title(title_str_input)

        if self.x_coord_min is not None:
            x_coord_min = np.minimum(x_coord_min, self.x_coord_min)
        if self.x_coord_max is not None:
            x_coord_max = np.maximum(x_coord_max, self.x_coord_max)
        if self.y_coord_min is not None:
            y_coord_min = np.minimum(y_coord_min, self.y_coord_min)
        if self.y_coord_max is not None:
            y_coord_max = np.maximum(y_coord_max, self.y_coord_max)


        self.ax.set(xlim=(x_coord_min - self.parameter.figure_additional_size,
                          x_coord_max + self.parameter.figure_additional_size),
                    ylim=(y_coord_min - self.parameter.figure_additional_size,
                          y_coord_max + self.parameter.figure_additional_size))
        if clear==False:
            self.x_coord_min,self.x_coord_max = x_coord_min, x_coord_max
            self.x_coord_min, self.x_coord_max = x_coord_min, x_coord_max
        # self.ax.legend(loc='upper center', ncol=3)

    def animate_frenet(self, i):
        # Measure execution time and relate it to animation speed
        time_from_lat_call = time_measure_get_diff()
        if not self.parameter.supress_warnings:
            if time_from_lat_call > 1 / self.parameter.animation_frames_per_sec:
                print("Animation plot is too slow ({:4.3f}s) for the requested frames per second ({}f/s)!".format(
                    time_from_lat_call,
                    self.parameter.animation_frames_per_sec))

        time.sleep(np.maximum(0, 1 / self.parameter.animation_frames_per_sec - time_from_lat_call))
        time_measure_start()

        # Plot road
        self.frenet_ax.clear()
        self.frenet_ax.axis('equal')
        plot_road_f(self.frenet_road, self.fig, self.frenet_ax, fast_plot=self.parameter.fast_animation)

        # create string for animation information
        title_str = ""
        t_current = 0
        x_coord_min, x_coord_max, y_coord_min, y_coord_max = 1e6, -1e6, 1e6, -1e6

        # iterate through vehicles
        for i_vehicle in range(self.n_vehicles):
            # Get data
            t_current = self.planning_data_container[i_vehicle].t_flat[i]
            xc_ego = CartesianTrajectory(self.planning_data_container[i_vehicle].x_flat[:, i])
            xf_ego = FrenetTrajectory(
                self.planning_data_container[i_vehicle].x_flat[:, i])  # self.road.transform_trajectory_f2c(xf_ego)
            x_ego = xc_ego.get_as_array()
            a_lat_ego = -x_ego[3] * x_ego[3] / self.vehicle_parameter[i_vehicle].length * np.tan(x_ego[4])
            alpha_ego = x_ego[2]
            x_center_ego = x_ego[0] + self.vehicle_parameter[i_vehicle].length / 2 * np.cos(alpha_ego)
            y_center_ego = x_ego[1] + self.vehicle_parameter[i_vehicle].length / 2 * np.sin(alpha_ego)

            title_str += r"v{}={:4.1f} ".format(i_vehicle, float(x_ego[3])) + r"$\frac{m}{s}$, "
            if not (self.parameter.animation_focus_ego and i_vehicle > 0):
                x_coord_min = np.minimum(x_coord_min, x_ego[0])
                x_coord_max = np.maximum(x_coord_max, x_ego[0])
                y_coord_min = np.minimum(y_coord_min, x_ego[1])
                y_coord_max = np.maximum(y_coord_max, x_ego[1])

            # Draw safety radius
            if not self.planner_parameter[i_vehicle].use_lifting:
                if i_vehicle in self.parameter.plot_safety_circles:
                    plot_safety_circles(self.frenet_ax, x_center_ego, y_center_ego,
                                        self.vehicle_parameter[i_vehicle].safety_radius,
                                        None,
                                        color=self.colors[i_vehicle], alpha_nom=0.2)

                    plot_covering_safety_circles(self.frenet_ax, self.frenet_road,
                                                 self.planner_parameter[i_vehicle],
                                                 xf_ego,
                                                 self.vehicle_parameter[i_vehicle],
                                                 color=self.colors[i_vehicle],
                                                 alpha=0.7)

            # Draw vehicle
            draw_vehicle_frenet(self.frenet_ax, self.road, xf_ego,
                                self.vehicle_parameter[i_vehicle].length,
                                self.vehicle_parameter[i_vehicle].chassis_length,
                                self.vehicle_parameter[i_vehicle].chassis_width,
                                self.parameter.animation_tire_length,
                                self.parameter.animation_tire_width,
                                fast_plot=self.parameter.fast_animation,
                                initial_guess_s=xf_ego.s[0])

            # Acceleration Arrows
            if self.planning_data_container[i_vehicle].u_flat.shape[0] > 0 and \
                    i_vehicle in self.parameter.plot_acceleration_arrows:
                u_ego = self.planning_data_container[i_vehicle].u_flat[:, i]
                # longitudinal acceleration arrow
                plot_arrow_vehicle_c(ax=self.frenet_ax, x_c=xc_ego, length=u_ego[0], angle_arrow=0, color="red",
                                     max_value=self.vehicle_parameter[i_vehicle].maximum_acceleration_force,
                                     min_value=-self.vehicle_parameter[i_vehicle].maximum_deceleration_force,
                                     scaling=4)

            # lateral acceleration arrow
            if i_vehicle in self.parameter.plot_acceleration_arrows:
                plot_arrow_vehicle_c(ax=self.frenet_ax, x_c=xc_ego, length=a_lat_ego, angle_arrow=np.pi / 2,
                                     color="red",
                                     max_value=self.vehicle_parameter[i_vehicle].maximum_lateral_acc,
                                     min_value=-self.vehicle_parameter[i_vehicle].maximum_lateral_acc,
                                     scaling=4)

            # Plot predictions
            if i_vehicle in self.parameter.plot_opp_predictions:
                for i_opponent in range(self.n_vehicles - 1):
                    if self.planning_data_container[i_vehicle].x_opp_predicted_all[i_opponent].shape[0] > 0:
                        traj_idx = int(
                            np.floor(i / self.planning_data_container[i_vehicle].n_trajectory_states_truncated))
                        x_pred = self.planning_data_container[i_vehicle].x_opp_predicted_all[i_opponent][:, :,
                                 traj_idx]
                        p_x = []
                        p_y = []
                        for index in range(0, x_pred.shape[1]):
                            xf = FrenetTrajectory(x_pred[:, index])
                            x = xf.get_as_array()  # self.road.transform_trajectory_f2c(xf).get_as_array()
                            x_center = x[0] + self.vehicle_parameter[1].length / 2 * np.cos(x[2])
                            y_center = x[1] + self.vehicle_parameter[1].length / 2 * np.sin(x[2])
                            p_x.append(x_center[0])
                            p_y.append(y_center[0])
                        self.frenet_ax.plot(p_x, p_y, alpha=0.6, zorder=-2, color="lightgrey", linewidth=10)

            # Plot plans
            if i_vehicle in self.parameter.plot_ego_plans:
                traj_idx = int(np.floor(i / self.planning_data_container[i_vehicle].n_trajectory_states_truncated))
                x_pred = self.planning_data_container[i_vehicle].x[:, :, traj_idx]
                p_x = []
                p_y = []
                colors = []
                for index in range(0, x_pred.shape[1]):
                    # Transform to cartesian
                    xf = FrenetTrajectory(x_pred[:, index])
                    x = xf.get_as_array()  # self.road.transform_trajectory_f2c(xf).get_as_array()
                    x_center = x[0] + self.vehicle_parameter[i_vehicle].length / 2 * np.cos(x[2])
                    y_center = x[1] + self.vehicle_parameter[i_vehicle].length / 2 * np.sin(x[2])

                    # Different color codings for ego plan
                    p_x.append(x_center[0])
                    p_y.append(y_center[0])
                    if self.parameter.planning_color_type == AnimationPlanningColorType.VELOCITY:
                        v_max = np.max([vehicle_p.maximum_velocity for vehicle_p in self.vehicle_parameter])
                        color_iter = (x[3] / v_max)[0]
                    elif self.parameter.planning_color_type == AnimationPlanningColorType.TIME:
                        color_iter = index / x_pred.shape[1]
                    else:
                        color_iter = self.colors[i_vehicle]
                    colors.append(color_iter)
                if self.parameter.planning_color_type == AnimationPlanningColorType.UNI:
                    self.frenet_ax.plot(p_x, p_y, alpha=0.6, zorder=-2, color=self.colors[i_vehicle], linewidth=10)
                else:
                    self.frenet_ax.scatter(p_x, p_y, alpha=0.6, zorder=-2, s=100, c=colors, cmap=self.velocity_cmap)

            if self.statistics is not None:
                idx_statistics = int(np.floor(t_current / self.statistics.delta_t))
                if self.statistics.collisions_follow[i_vehicle, idx_statistics] > 0:
                    draw_vehicle(self.frenet_ax, xc_ego,
                                 self.vehicle_parameter[i_vehicle].length,
                                 self.vehicle_parameter[i_vehicle].chassis_length,
                                 self.vehicle_parameter[i_vehicle].chassis_width,
                                 self.parameter.animation_tire_length,
                                 self.parameter.animation_tire_width,
                                 fast_plot=True,
                                 color="red")
                if self.statistics.collisions_lead[i_vehicle, idx_statistics] > 0:
                    draw_vehicle(self.frenet_ax, xc_ego,
                                 self.vehicle_parameter[i_vehicle].length,
                                 self.vehicle_parameter[i_vehicle].chassis_length,
                                 self.vehicle_parameter[i_vehicle].chassis_width,
                                 self.parameter.animation_tire_length,
                                 self.parameter.animation_tire_width,
                                 fast_plot=True,
                                 color="green")

        # set title and zoom
        title_str = r"$t_{real}$ = " + "{:.2f}s, ".format(get_relative_time()) + \
                    r"$t_{sim}$ = " + "{:.2f}s, ".format(t_current) + title_str
        if self.parameter.animation_pannels == AnimationPanels.FRENET:
            self.frenet_ax.set_title(title_str)
        else:
            self.frenet_ax.set_title("Frenet Coordinate Frame")
            self.frenet_ax.set_xlabel("s (m)")
            self.frenet_ax.set_ylabel("n (m)")
        self.frenet_ax.set(xlim=(x_coord_min - self.parameter.figure_additional_size,
                                 x_coord_max + self.parameter.figure_additional_size),
                           ylim=(y_coord_min - self.parameter.figure_additional_size,
                                 y_coord_max + self.parameter.figure_additional_size))

        # self.ax.legend(loc='upper center', ncol=3)


def _plot_to_rgbarray(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='jpeg', dpi=100)
    buf.seek(0)
    image = Image.open(buf)
    return np.array(image)
