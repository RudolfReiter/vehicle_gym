from typing import TYPE_CHECKING, List
import matplotlib.cm as cm
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from vehiclegym.utils.automotive_datastructures import (CartesianTrajectory,
                                                        FrenetTrajectory)

if TYPE_CHECKING:
    from vehiclegym.road.road import Road
    from vehiclegym.planner.trajectory_planner_base import PlanningDataContainer


class LineDataUnits(Line2D):
    def __init__(self, *args, **kwargs):
        _lw_data = kwargs.pop("linewidth", 1)
        super().__init__(*args, **kwargs)
        self._lw_data = _lw_data

    def _get_lw(self):
        if self.axes is not None:
            ppd = 72. / self.axes.figure.dpi
            trans = self.axes.transData.transform
            return ((trans((1, self._lw_data)) - trans((0, 0))) * ppd)[1]
        else:
            return 1

    def _set_lw(self, lw):
        self._lw_data = lw

    _linewidth = property(_get_lw, _set_lw)


def plot_colourline_2(ax, x, y, c, alpha: float = 1, c_ref=None, plot_c_bar=True, label="Velocity"):
    if plot_c_bar:
        sm = plt.cm.ScalarMappable(cmap=plt.cm.cool, norm=plt.Normalize(vmin=np.min(c_ref), vmax=np.max(c_ref)))
        sm._A = []
        plt.colorbar(sm, ax=ax, label=label)
    if c_ref is not None:
        c = cm.cool((c - np.min(c_ref)) / np.maximum((np.max(c_ref) - np.min(c_ref)), 0.1))
    else:
        c = cm.cool((c - np.min(c)) / np.maximum((np.max(c) - np.min(c)), 0.1))

    for i in np.arange(len(x) - 1):
        ax.plot([x[i], x[i + 1]], [y[i], y[i + 1]], c=c[i], alpha=alpha)
    return

def plot_colourline_3(ax, x, y, c, alpha: float = 1, c_ref=None, plot_c_bar=True, label=r"Velocity $(\frac{\mathrm{m}}/{\mathrm{s}})$"):
    if plot_c_bar:
        sm = plt.cm.ScalarMappable(cmap=plt.cm.hot, norm=plt.Normalize(vmin=np.min(c_ref), vmax=np.max(c_ref)))
        sm._A = []
        plt.colorbar(sm, ax=ax, label=label, location='top',  fraction=0)
    if c_ref is not None:
        c = cm.hot((c - np.min(c_ref)) / np.maximum((np.max(c_ref) - np.min(c_ref)), 0.1))
    else:
        c = cm.hot((c - np.min(c)) / np.maximum((np.max(c) - np.min(c)), 0.1))

    for i in np.arange(len(x) - 1):
        ax.plot([x[i], x[i + 1]], [y[i], y[i + 1]], c=c[i], alpha=alpha)
    return


def plot_road_curvature(road: "Road"):
    fig, axs = plt.subplots(1, 1)
    axs.plot(road.s_grid_, road.kappa_grid_)
    plt.title("Curvature")
    axs.set_xlabel("Path length s in [m/s]")
    axs.set_ylabel("Curvature kappa in [rad/m]")
    fig.tight_layout()
    axs.grid()
    plt.show()


def plot_road(road: "Road", fig=None, axs=None, fast_plot: bool = False):
    outer_scope = False
    if axs is None:
        fig, axs = plt.subplots(1, 1)
        outer_scope = True
    if not fast_plot:
        axs.plot(road.x_, road.y_, "--", label="center line", zorder=-20, color="lightgrey", linewidth=3)
        line = LineDataUnits(road.x_, road.y_, zorder=-21, color="darkgrey", linewidth=road.road_options_.road_width, alpha=0.85)
        axs.add_line(line)
        line2 = LineDataUnits(road.left_border_trajectory_c.x, road.left_border_trajectory_c.y, linestyle='dashed',
                              color='lightgrey', label="left border", zorder=-1, linewidth=0.5)
        axs.add_line(line2)
        line3 = LineDataUnits(road.right_border_trajectory_c.x, road.right_border_trajectory_c.y, linestyle='dashed',
                              color='lightgrey', label="right border", zorder=-1, linewidth=0.5)
        axs.add_line(line3)
    line2 = LineDataUnits(road.left_border_trajectory_c.x, road.left_border_trajectory_c.y, color='dimgrey',
                          zorder=-2, linewidth=0.5)
    axs.add_line(line2)
    line3 = LineDataUnits(road.right_border_trajectory_c.x, road.right_border_trajectory_c.y, color='dimgrey',
                          zorder=-2, linewidth=0.5)
    axs.add_line(line3)

    axs.axis('equal')
    plt.title("Road layout")
    axs.set_xlabel(r"x (m)")
    axs.set_ylabel(r"y (m)")
    fig is not None and fig.tight_layout()
    # axs.grid()
    if outer_scope:
        plt.legend()
        plt.show()


def plot_road_f(road: "Road", fig=None, axs=None, fast_plot: bool = False):
    outer_scope = False
    if fig is None:
        fig, axs = plt.subplots(1, 1)
        outer_scope = True
    if not fast_plot:
        axs.plot(road.s_grid_, np.zeros_like(road.s_grid_), "--", label="center line", zorder=-20, color="lightgrey",
                 linewidth=3)
        # line = LineDataUnits(road.x_, road.y_, zorder=-21, color="black", linewidth=road.road_options_.road_width, alpha=0.85)
        # axs.add_line(line)
        line2 = LineDataUnits(road.s_grid_, road.nl_grid_, linestyle='dashed',
                              color='lightgrey', label="left border", zorder=-1, linewidth=0.5)
        axs.add_line(line2)
        line3 = LineDataUnits(road.s_grid_, -road.nr_grid_, linestyle='dashed',
                              color='lightgrey', label="right border", zorder=-1, linewidth=0.5)
        axs.add_line(line3)
    line2 = LineDataUnits(road.s_grid_, road.nl_grid_, color='dimgrey',
                          zorder=-2, linewidth=0.5)
    axs.add_line(line2)
    line3 = LineDataUnits(road.s_grid_, -road.nr_grid_, color='dimgrey',
                          zorder=-2, linewidth=0.5)
    axs.add_line(line3)

    axs.axis('equal')
    plt.title("Road layout")
    axs.set_xlabel(r"s (m)")
    axs.set_ylabel(r"n (m)")
    fig.tight_layout()
    # axs.grid()
    if outer_scope:
        plt.legend()
        plt.show()


def plot_c_trajectory_on_road(road: "Road", trajectory_c: CartesianTrajectory, axs=None, fig=None):
    show_plots = False
    if axs is None:
        fig, axs = plt.subplots(1, 1)
        show_plots = True
    plot_road(road, fig, axs)
    plt.plot(trajectory_c.x, trajectory_c.y, 'g', label="trajectory")
    if show_plots:
        plt.legend()
        plt.show()


def plot_f_trajectory_on_road(road: "Road", trajectory_f: FrenetTrajectory,
                              zoom_in: bool = False,
                              plot_velocity: bool = False,
                              axs=None,
                              fig=None,
                              label="trajectory",
                              color='g') -> None:
    show_plots = False
    if axs is None:
        fig, axs = plt.subplots(1, 1)
        show_plots = True

    plot_road(road, fig, axs)
    trajectory_c = road.transform_trajectory_f2c(trajectory_f)
    if plot_velocity:
        plot_colourline_2(axs, trajectory_c.x, trajectory_c.y, trajectory_c.v)
    else:
        plt.plot(trajectory_c.x, trajectory_c.y, color=color, label=label)
    if zoom_in:
        plt.xlim(np.min(trajectory_c.x - 5), np.max(trajectory_c.x + 5))
        plt.ylim(np.min(trajectory_c.y - 5), np.max(trajectory_c.y + 5))
    if show_plots:
        plt.legend()
        plt.show()


def plot_f_trajectories_on_road(road: "Road",
                                trajectories_f: List[FrenetTrajectory],
                                circle_shapes: List[float] = None) -> None:
    fig, axs = plt.subplots(1, 1)
    plot_road(road, fig, axs)

    color_map = plt.get_cmap('winter')
    colors_set = [color_map(i) for i in np.linspace(0, 1, len(trajectories_f[0].s))]

    for trajectory_idx, trajectory_f in enumerate(trajectories_f):
        str_name = "trajectory " + str(trajectory_idx)
        trajectory_c = road.transform_trajectory_f2c(trajectory_f)
        if circle_shapes is None:
            plt.plot(trajectory_c.x, trajectory_c.y, label=str_name)
        else:
            for point_idx, (coord_x, coord_y) in enumerate(zip(trajectory_c.x, trajectory_c.y)):
                circle = plt.Circle((coord_x, coord_y),
                                    circle_shapes[trajectory_idx],
                                    fill=False,
                                    color=colors_set[point_idx])
                axs.add_patch(circle)

    plt.legend()
    plt.show()


class RolloutTrajectoryPlotter:
    def __init__(self, road: "Road", v_min: float = 0, v_max: float = 50):
        self.road = road
        self.v_min = v_min
        self.v_max = v_max
        self.fig = None
        self.axs = None

    def plot_trajectories(self, ego_trajectory: "PlanningDataContainer",
                          opp_trajectory: "PlanningDataContainer" = None):
        self.fig, self.axs = plt.subplots(1, 1)
        plot_road(self.road, self.fig, self.axs)
        first_call = True
        for idx in range(ego_trajectory.x.shape[2]):
            trajectory_main = FrenetTrajectory(ego_trajectory.x[:, :, idx])
            trajectory_rollout = FrenetTrajectory(ego_trajectory.x_ro[:, :, idx])
            trajectory_c = self.road.transform_trajectory_f2c(trajectory_main)
            trajectory_ro_c = self.road.transform_trajectory_f2c(trajectory_rollout)

            plot_colourline_2(self.axs,
                              trajectory_c.x,
                              trajectory_c.y,
                              trajectory_c.v,
                              c_ref=[0, self.v_max],
                              plot_c_bar=first_call)
            first_call = False

            plt.plot(trajectory_c.x, trajectory_c.y, 'o', markersize=2, color="black", alpha=0.8)
            plt.plot(trajectory_c.x[0], trajectory_c.y[0], 'o', markersize=5, color="black")
            plot_colourline_2(self.axs,
                              trajectory_ro_c.x,
                              trajectory_ro_c.y,
                              trajectory_ro_c.v,
                              c_ref=[0, self.v_max],
                              plot_c_bar=False,
                              alpha=0.7)
        if opp_trajectory is not None:
            for idx in range(opp_trajectory.x.shape[2]):
                trajectory_opponent = FrenetTrajectory(opp_trajectory.x[:, :, idx])
                trajectory_opp_c = self.road.transform_trajectory_f2c(trajectory_opponent)
                plot_colourline_2(self.axs,
                                  trajectory_opp_c.x,
                                  trajectory_opp_c.y,
                                  trajectory_opp_c.v,
                                  c_ref=[0, self.v_max],
                                  plot_c_bar=False,
                                  alpha=0.7)
                plt.plot(trajectory_opp_c.x, trajectory_opp_c.y, 'o', markersize=2, color="darkred", alpha=0.8)
                plt.plot(trajectory_opp_c.x[0], trajectory_opp_c.y[0], 'o', markersize=5, color="darkred")
        plt.show()
