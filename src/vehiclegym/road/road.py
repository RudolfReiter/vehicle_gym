import enum
import pickle
import tempfile
import warnings
from copy import copy, deepcopy
from dataclasses import dataclass, field
from time import time
from timeit import timeit
from typing import List
from dataclasses_json import dataclass_json
import casadi as cs
import numpy as np
from scipy import interpolate
from vehiclegym.road.road_helpers import compute_curvature, integrate_curvature, FrozenClass
from vehiclegym.utils.automotive_datastructures import CartesianTrajectory, FrenetTrajectory
from vehiclegym.road.curvature_generator import (
    generate_boundary,
    generate_random,
    generate_circle,
)
from vehiclegym.plotting.plotters import (
    plot_c_trajectory_on_road,
    plot_colourline_2,
    plot_f_trajectories_on_road,
    plot_f_trajectory_on_road,
    plot_road,
    plot_road_curvature,
)
from vehiclegym.road.curvature_generator import RandomRoadParameters


@dataclass_json
@dataclass
class RoadOptions(FrozenClass):
    delta_s: float = 4  # spacing between discrete road points
    n_points: int = 200  # number of points
    start_point_cartesian: List[float] = field(default_factory=lambda: [0, 0, 0, 0, 0])
    random_road_parameters: RandomRoadParameters = field(
        default_factory=lambda: RandomRoadParameters()
    )

    @property
    def road_width(self):
        return 2*self.random_road_parameters.boundary_mu



class Road:
    def __init__(
            self,
            road_options: RoadOptions,
            kappa: np.ndarray = None,
            s: np.ndarray = None,
            p_xy: np.ndarray = None,
            phi: np.ndarray = None,
            nl: np.ndarray = None,
            nr: np.ndarray = None,
    ):
        """
        Initialise class either with just random road parameters or with exact values
        :param road_options: options for generating random roads
        :param kappa: curvature
        :param s: path length
        :param p_xy: cartesian points x and y
        :param phi: tangent angle
        :param nl: distance left
        :param nr: distance right
        """
        assert (kappa is not None and s is not None and p_xy is not None and
                phi is not None and nl is not None and nr is not None) or \
               (kappa is None and s is None and p_xy is None and
                phi is None and nl is None and nr is None)

        is_initialized = (kappa is not None)

        self.left_border_trajectory_c_ = None
        self.right_border_trajectory_c_ = None
        self.left_border_trajectory_f_ = None
        self.right_border_trajectory_f_ = None
        self.spline_s2kappa = None
        self.spline_s2phi_ = None
        self.spline_s2y_ = None
        self.spline_s2x_ = None

        self.s_grid_ = copy(s)
        self.kappa_grid_ = copy(kappa)
        self.nl_grid_ = copy(nl)
        self.nr_grid_ = copy(nr)
        if p_xy is not None:
            self.x_grid_ = p_xy[:, 0]
            self.y_grid_ = p_xy[:, 1]
        else:
            self.x_grid_ = None
            self.y_grid_ = None
        self.phi_grid_ = phi

        self.road_options_ = road_options
        self.opti = cs.Opti()

        # optimization problem to find point on trajectory
        if self.s_grid_ is not None:
            n_grid = self.s_grid_.__len__()
            s_grid_opti = self.s_grid_
        else:
            n_grid = road_options.n_points
            s_grid_opti = np.linspace(
                0,
                road_options.delta_s * (road_options.n_points - 1),
                road_options.n_points,
            )

        self.par_x_grid_ = self.opti.parameter(n_grid)
        self.par_y_grid_ = self.opti.parameter(n_grid)
        s2x = cs.interpolant("interp_s2x", "bspline", [s_grid_opti])
        s2y = cs.interpolant("interp_s2y", "bspline", [s_grid_opti])
        options = {
            "print_time": 0,
            "ipopt": {
                "max_iter": 1e4,
                "print_level": 0,
                "acceptable_tol": 1e-3,
                "acceptable_obj_change_tol": 1e-3,
            },
        }
        self.opti.solver("ipopt", options)
        self.cartesian_x_ = self.opti.parameter(1)
        self.cartesian_y_ = self.opti.parameter(1)
        self.pos_s_ = self.opti.variable(1)
        pos_x = s2x(self.pos_s_, self.par_x_grid_)
        pos_y = s2y(self.pos_s_, self.par_y_grid_)
        self.opti.minimize(
            (pos_x - self.cartesian_x_) ** 2 + (pos_y - self.cartesian_y_) ** 2
        )

        if not is_initialized:
            self.randomize()
        else:
            self.post_initialize()

    @classmethod
    def from_curvature(
            cls,
            road_options: RoadOptions,
            s: np.ndarray,
            kappa: np.ndarray,
            nl: np.ndarray,
            nr: np.ndarray,
    ):
        """
        Initialise class from curvature information. Cartesian points are computed
        :return: Initialised class
        """
        road_options = copy(road_options)
        road_options.n_points = s.__len__()
        road_options.delta_s = s[1] - s[0]

        x, y, phi = integrate_curvature(s_grid=s, kappa_grid=kappa)
        x = np.expand_dims(x, axis=1)
        y = np.expand_dims(y, axis=1)
        p_xy = np.hstack((x, y))
        return cls(road_options, kappa=kappa, s=s, p_xy=p_xy, phi=phi, nl=nl, nr=nr)

    @classmethod
    def from_xy(
            cls,
            road_options: RoadOptions,
            p_xy: np.ndarray,
            nl: np.ndarray,
            nr: np.ndarray,
            s: np.ndarray = None,
            smoothing_par=(11, 3),
            do_plot: bool = False,
    ):
        """
        Initialise class from cartesian points information. Cartesian points are computed
        :return: Initialised class
        """
        kappa, s_lin, phi = compute_curvature(
            p_xy, filter_coefs=smoothing_par, do_plot=do_plot
        )
        if s is None:
            s = s_lin

        road_options = copy(road_options)
        road_options.n_points = nl.__len__()
        road_options.delta_s = s[1] - s[0]
        return cls(road_options, kappa=kappa, s=s, p_xy=p_xy, phi=phi, nl=nl, nr=nr)

    def update_from_xy(
            self, p_xy: np.ndarray, nl: np.ndarray, nr: np.ndarray, s: np.ndarray = None
    ):
        assert nl.__len__() == p_xy.shape[0] and nl.__len__() == nr.__len__()
        self.left_border_trajectory_c_ = None
        self.right_border_trajectory_c_ = None
        self.left_border_trajectory_f_ = None
        self.right_border_trajectory_f_ = None

        self.kappa_grid_, s_lin, self.phi_grid_ = compute_curvature(p_xy)
        if s is None:
            s = s_lin
        self.s_grid_ = s
        self.x_grid_ = p_xy[:, 0]
        self.y_grid_ = p_xy[:, 1]
        self.nl_grid_ = nl
        self.nr_grid_ = nr

        self.post_initialize()

    def set_kappa(self, kappa_grid):
        assert len(kappa_grid) == len(self.kappa_grid_)
        self.kappa_grid_ = kappa_grid
        self.post_initialize()

    def set_bounds(self, nl_grid, nr_grid):
        assert len(nl_grid) == len(self.nl_grid_)
        assert len(nr_grid) == len(self.nr_grid_)
        self.nl_grid_ = nl_grid
        self.nr_grid_ = nr_grid
        self.post_initialize()

    def clip_s(self, s):
        return np.clip(s, 0, self.spline_s2kappa.x[-1])

    def post_initialize(self):
        # set opti parameters
        self.opti.set_value(self.par_x_grid_, self.x_grid_)
        self.opti.set_value(self.par_y_grid_, self.y_grid_)

        # define splines
        self.spline_s2x_ = interpolate.interp1d(self.s_grid_, self.x_grid_)
        self.spline_s2y_ = interpolate.interp1d(self.s_grid_, self.y_grid_)
        self.spline_s2phi_ = interpolate.interp1d(
            self.s_grid_, np.unwrap(self.phi_grid_)
        )
        self.spline_s2kappa = interpolate.interp1d(
            self.s_grid_, self.kappa_grid_, bounds_error=False, fill_value=0
        )
        self.spline_s2nlx_ = interpolate.interp1d(
            self.s_grid_,
            self.x_grid_ + np.cos(self.phi_grid_ + np.pi / 2) * self.nl_grid_,
        )
        self.spline_s2nly_ = interpolate.interp1d(
            self.s_grid_,
            self.y_grid_ + np.sin(self.phi_grid_ + np.pi / 2) * self.nl_grid_,
        )
        self.spline_s2nrx_ = interpolate.interp1d(
            self.s_grid_,
            self.x_grid_ - np.cos(self.phi_grid_ + np.pi / 2) * self.nr_grid_,
        )
        self.spline_s2nry_ = interpolate.interp1d(
            self.s_grid_,
            self.y_grid_ - np.sin(self.phi_grid_ + np.pi / 2) * self.nr_grid_,
        )
        self.spline_s2nr_ = interpolate.interp1d(
            self.s_grid_, self.nr_grid_, bounds_error=False, fill_value=10
        )
        self.spline_s2nl_ = interpolate.interp1d(
            self.s_grid_, self.nl_grid_, bounds_error=False, fill_value=10
        )

    @property
    def right_border_trajectory_f(self):
        """
        Right border trajectory in frenet coordinates
        Compute boundary trajectories on demand
        """
        if self.right_border_trajectory_f_ is None:
            border_trajectory = FrenetTrajectory()
            border_trajectory.set_as_array(np.zeros((5, len(self.s_grid_))))
            border_trajectory.s = copy(self.s_grid_)
            border_trajectory.n = -self.nr_grid_
            self.right_border_trajectory_f_ = border_trajectory
        return self.right_border_trajectory_f_

    @property
    def right_border_trajectory_c(self):
        """
        :return: Right border trajectory in cartesian coordinates
        """
        if self.right_border_trajectory_c_ is None:
            self.right_border_trajectory_c_ = self.transform_trajectory_f2c(
                self.right_border_trajectory_f
            )
        return self.right_border_trajectory_c_

    @property
    def left_border_trajectory_f(self):
        """
        :return: Left border trajectory in frenet coordinates
        """
        if self.left_border_trajectory_f_ is None:
            border_trajectory = FrenetTrajectory()
            border_trajectory.set_as_array(np.zeros((5, len(self.s_grid_))))
            border_trajectory.s = copy(self.s_grid_)
            border_trajectory.n = self.nl_grid_
            self.left_border_trajectory_f_ = border_trajectory
        return self.left_border_trajectory_f_

    @property
    def left_border_trajectory_c(self):
        """
        :return: Left border trajectory in cartesian coordinates
        """
        if self.left_border_trajectory_c_ is None:
            self.left_border_trajectory_c_ = self.transform_trajectory_f2c(
                self.left_border_trajectory_f
            )
        return self.left_border_trajectory_c_

    def randomize(self, seed: int = None):
        if seed is not None:
            self.road_options_.random_road_parameters.seed = seed
        self.left_border_trajectory_c_ = None
        self.right_border_trajectory_c_ = None
        self.left_border_trajectory_f_ = None
        self.right_border_trajectory_f_ = None
        self.road_options_.random_road_parameters.delta_s = self.road_options_.delta_s
        self.road_options_.random_road_parameters.number_points = (
            self.road_options_.n_points
        )
        self.s_grid_, self.kappa_grid_ = generate_random(
            self.road_options_.random_road_parameters
        )
        self.nl_grid_, self.nr_grid_ = generate_boundary(
            self.road_options_.random_road_parameters
        )

        self.x_grid_, self.y_grid_, self.phi_grid_ = integrate_curvature(
            s_grid=self.s_grid_, kappa_grid=self.kappa_grid_
        )
        self.post_initialize()

    def transform_trajectory_f2c(
            self, trajectory_f: FrenetTrajectory
    ) -> CartesianTrajectory:
        # Check the proper range of the s-value
        if hasattr(self, "track_len"):
            trajectory_f.s = np.mod(trajectory_f.s, self.track_len)
        if hasattr(trajectory_f.s, "__len__"):
            assert trajectory_f.s[-1] <= self.s_grid_[-1]

            if trajectory_f.s[0] < self.s_grid_[0]:
                warnings.warn(
                    "Transform trajectory values out of range! Clipping values."
                )
                for i, s in enumerate(trajectory_f.s):
                    if s < self.s_grid_[i]:
                        trajectory_f.s[i] = copy(self.s_grid_[i])
            if trajectory_f.s[0] < self.s_grid_[0]:
                warnings.warn(
                    "Transform trajectory values out of range (grid)! Clipping values."
                )
                for i, s in enumerate(trajectory_f.s_grid_):
                    if s < self.s_grid_[i]:
                        trajectory_f.s[i] = copy(self.s_grid_[i])

        # Compute cartesian trajectory by the addition of a base vector to the center line spline plus a perpendicular
        # vector with the lateral distance state "n"
        trajectory_c = CartesianTrajectory()
        cartesian_total_angle = self.spline_s2phi_(trajectory_f.s) + np.pi / 2
        trajectory_c.x = self.spline_s2x_(trajectory_f.s) + trajectory_f.n * np.cos(
            cartesian_total_angle
        )
        trajectory_c.y = self.spline_s2y_(trajectory_f.s) + trajectory_f.n * np.sin(
            cartesian_total_angle
        )
        trajectory_c.phi = self.spline_s2phi_(trajectory_f.s) + trajectory_f.alpha

        # velocity and steering angle are constant
        trajectory_c.v = copy(trajectory_f.v)
        trajectory_c.delta = copy(trajectory_f.delta)

        return trajectory_c

    def transform_trajectory_c2f_fast(
            self, trajectory_c: CartesianTrajectory, initial_guess_s: float = 0
    ) -> FrenetTrajectory:
        dx = self.x_ - trajectory_c.x
        dy = self.y_ - trajectory_c.y
        dist = dx ** 2 + dy ** 2
        idx_s = np.argmin(dist)
        trajectory_f = FrenetTrajectory()
        trajectory_f.s = self.s_grid_[idx_s]

        # Computing n and its direction requires the cross product of finding the side of the point
        trajectory_f.n = np.sqrt((trajectory_c.x - self.spline_s2x_(trajectory_f.s)) ** 2 +
                                 (trajectory_c.y - self.spline_s2y_(trajectory_f.s)) ** 2)
        epsilon = 1e-4
        dx = self.spline_s2x_(trajectory_f.s + epsilon) - self.spline_s2x_(trajectory_f.s)
        dy = self.spline_s2y_(trajectory_f.s + epsilon) - self.spline_s2y_(trajectory_f.s)
        px = trajectory_c.x - self.spline_s2x_(trajectory_f.s)
        py = trajectory_c.y - self.spline_s2y_(trajectory_f.s)
        isleft = ((dx * py - dy * px) > 0)
        if not isleft:
            trajectory_f.n *= -1
        trajectory_f.alpha = trajectory_c.phi - self.spline_s2phi_(trajectory_f.s)
        trajectory_f.v = copy(trajectory_c.v)
        trajectory_f.delta = copy(trajectory_c.delta)

        return trajectory_f

    def transform_trajectory_c2f(
            self, trajectory_c: CartesianTrajectory, initial_guess_s: float = 0
    ) -> FrenetTrajectory:
        """
        Right now only working for one point in trajectory
        :param trajectory_c:
        :param initial_guess_s:
        :return:
        """

        self.opti.set_value(self.cartesian_x_, trajectory_c.x)
        self.opti.set_value(self.cartesian_y_, trajectory_c.y)
        self.opti.set_initial(self.pos_s_, initial_guess_s)
        sol = self.opti.solve()
        trajectory_f = FrenetTrajectory()
        trajectory_f.s = np.maximum(sol.value(self.pos_s_), 1e-6)

        # Computing n and its direction requires the cross product of finding the side of the point
        trajectory_f.n = np.sqrt((trajectory_c.x - self.spline_s2x_(trajectory_f.s)) ** 2 +
                                 (trajectory_c.y - self.spline_s2y_(trajectory_f.s)) ** 2)
        epsilon = 1e-4
        dx = self.spline_s2x_(trajectory_f.s + epsilon) - self.spline_s2x_(trajectory_f.s)
        dy = self.spline_s2y_(trajectory_f.s + epsilon) - self.spline_s2y_(trajectory_f.s)
        px = trajectory_c.x - self.spline_s2x_(trajectory_f.s)
        py = trajectory_c.y - self.spline_s2y_(trajectory_f.s)
        isleft = ((dx * py - dy * px) > 0)
        if not isleft:
            trajectory_f.n *= -1
        trajectory_f.alpha = trajectory_c.phi - self.spline_s2phi_(trajectory_f.s)
        trajectory_f.v = copy(trajectory_c.v)
        trajectory_f.delta = copy(trajectory_c.delta)

        return trajectory_f

    @property
    def x_(self):
        return self.x_grid_

    @property
    def y_(self):
        return self.y_grid_

    @property
    def phi_(self):
        return self.phi_grid_

    @property
    def s_(self):
        return self.s_grid_

    def __getstate__(self):
        return {
            "road_options_": self.road_options_,
            "s_grid_": self.s_grid_,
            "x_grid_": self.x_grid_,
            "y_grid_": self.y_grid_,
            "kappa_grid_": self.kappa_grid_,
            "nl_grid_": self.nl_grid_,
            "nr_grid_": self.nr_grid_,
            "phi_grid_": self.phi_grid_,
        }

    def __setstate__(self, state):
        self.road_options_ = state["road_options_"]
        self.s_grid_ = state["s_grid_"]
        self.nl_grid_ = state["nl_grid_"]
        self.nr_grid_ = state["nr_grid_"]
        self.kappa_grid_ = state["kappa_grid_"]
        self.x_grid_ = state["x_grid_"]
        self.y_grid_ = state["y_grid_"]
        self.phi_grid_ = state["phi_grid_"]
        self.left_border_trajectory_c_ = None
        self.right_border_trajectory_c_ = None
        self.left_border_trajectory_f_ = None
        self.right_border_trajectory_f_ = None
        self.opti = cs.Opti()
        if self.s_grid_ is not None:
            n_grid = self.s_grid_.__len__()
        else:
            n_grid = self.road_options_.n_points
        self.par_x_grid_ = self.opti.parameter(n_grid)
        self.par_y_grid_ = self.opti.parameter(n_grid)
        self.post_initialize()


class CircularRoad(Road):
    def __init__(self, road: Road, s_add_neg: float, s_add_pos: float, smooth_par = None, do_plot=False):
        # check if last point is exactly the first point
        assert s_add_neg >= 0 and s_add_pos >= 0

        self.base_road = road
        self.s_add_neg = s_add_neg
        self.s_add_pos = s_add_pos

        x_grid = deepcopy(road.x_grid_)
        y_grid = deepcopy(road.y_grid_)
        nl_grid = deepcopy(road.nl_grid_)
        nr_grid = deepcopy(road.nr_grid_)
        s_grid = deepcopy(road.s_grid_)

        dist_last_point = np.sqrt((road.x_grid_[-1] - road.x_grid_[0]) ** 2 + (road.y_grid_[-1] - road.y_grid_[0]) ** 2)
        if dist_last_point > 1e-3:
            first_idx = 0
            track_len = s_grid[-1] + dist_last_point
        else:
            first_idx = 1
            track_len = s_grid[-1]

        self.track_len = track_len
        self.s_switch = track_len / 2

        idx_front_add = np.argmin(np.abs(s_grid - s_add_pos))
        idx_back_add = np.argmin(np.abs(s_grid - (track_len - s_add_neg)))

        all_grids = [x_grid, y_grid, nl_grid, nr_grid]
        for i, grid in enumerate(all_grids):
            all_grids[i] = np.hstack((grid[idx_back_add:], grid[first_idx:], grid[first_idx:idx_front_add]))
        [x_grid, y_grid, nl_grid, nr_grid] = all_grids
        s_grid_pre = s_grid[idx_back_add:] - track_len
        s_grid_post = s_grid[first_idx:idx_front_add] + track_len
        s_grid = np.hstack((s_grid_pre, s_grid[first_idx:], s_grid_post))

        x = np.expand_dims(x_grid, axis=1)
        y = np.expand_dims(y_grid, axis=1)
        p_xy = np.hstack((x, y))

        road_options_ = copy(road.road_options_)
        road_options_.n_points = s_grid.__len__()
        road_options_.delta_s = s_grid[1] - s_grid[0]
        if smooth_par is None:
            smooth_par = (51, 3)

        tmp_road = Road.from_xy(road_options=road_options_, p_xy=p_xy, nl=nl_grid, nr=nr_grid, s=s_grid,
                                smoothing_par=smooth_par, do_plot=do_plot)

        x = np.expand_dims(tmp_road.x_grid_, axis=1)
        y = np.expand_dims(tmp_road.y_grid_, axis=1)
        p_xy = np.hstack((x, y))
        super().__init__(road_options=road_options_, kappa=tmp_road.kappa_grid_, s=tmp_road.s_grid_,
                         p_xy=p_xy, phi=tmp_road.phi_grid_, nl=tmp_road.nl_grid_, nr=tmp_road.nr_grid_)

    def project(self, s_ref: float, s_input) -> [float, float, bool]:
        """
        Projects an s_input to the right part of the track which is needed due circumvention issues. Also modulates
        the reference value
        :param s_ref: reference s, which is the ego s position
        :param s_input: testing s, which should be projected
        :return:
            s_ref_out, modulated reference s
            s_output, projected opponent s
            modulated, binary variable indicating modulation
        """
        s_ref_out = np.mod(s_ref, self.track_len)
        modulated = False
        if np.abs(s_ref - s_ref_out) > 1e-3:
            modulated = True
        s_output = copy(s_input)
        if s_ref_out < self.s_switch:  # first part of the track
            if s_input > (self.track_len - self.s_add_neg):
                s_output -= self.track_len
        else:  # second part of the track
            if s_input < self.s_add_pos:
                s_output += self.track_len

        return s_ref_out, s_output, modulated

    def project_states(self, ego_state: np.ndarray, opp_states: List[np.ndarray]):
        s0 = ego_state[0]
        if opp_states is not None and len(opp_states) > 0:
            s0_opps = [opp[0] for opp in opp_states]
            modulated = False
            for i in range(len(s0_opps)):
                ego_state[0], opp_states[i][0], modulated_tmp = self.project(s0, s0_opps[i])
                modulated = modulated or modulated_tmp
        else:
            ego_state[0], _, modulated = self.project(s0, 0)
        return ego_state, opp_states, modulated


if __name__ == "__main__":
    print("Test road generation")
    # Generate road options
    road_options_local = RoadOptions()
    road_options_local.random_road_parameters.boundary_mu = 6
    road_options_local.random_road_parameters.boundary_std = 6

    # create road from cartesian
    N = 100
    phi = np.linspace(0, 2 * np.pi, N)
    R = 50
    x = np.cos(phi) * R
    y = np.sin(phi) * R
    x = np.expand_dims(x, axis=1)
    y = np.expand_dims(y, axis=1)
    n_l = 5 * np.ones((N,))
    n_r = 5 * np.ones((N,))
    p_xy = np.hstack((x, y))
    road1 = Road.from_xy(road_options=road_options_local, p_xy=p_xy, nl=n_l, nr=n_r)

    road3 = CircularRoad(road1, s_add_neg=100, s_add_pos=100)
    plot_road(road3)

    plot_road(road1)
    road1.randomize()
    plot_road(road1)
    road1.update_from_xy(p_xy=p_xy, nl=n_l, nr=n_r)
    plot_road(road1)

    # create road from frenet
    s = np.linspace(0, 100, N)
    kappa = np.ones_like(s) * 0.01
    road2 = Road.from_curvature(
        road_options=road_options_local, s=s, kappa=kappa, nl=n_l, nr=n_r
    )
    plot_road(road2)

    # Generate initial state
    cartesian_state0 = CartesianTrajectory()

    # Create road_local with options
    road_local = Road(road_options_local)

    # Generate exemplary trajectory
    example_trajectory_f = copy(road_local.left_border_trajectory_f)
    example_trajectory_f.n -= 2

    # Transform a cartesian point
    test_point_c = CartesianTrajectory()
    test_point_c.x = 10
    test_point_c.y = 5
    test_point_f = road_local.transform_trajectory_c2f(
        trajectory_c=test_point_c, initial_guess_s=1
    )
    print(
        "Transformed coordinates: s: {}, n: {}".format(test_point_f.s, test_point_f.n)
    )

    # Plot the road and the curvature
    plot_road_curvature(road_local)
    plot_road(road_local)
    plot_f_trajectory_on_road(road=road_local, trajectory_f=example_trajectory_f)

    road2 = deepcopy(road_local)

    # Test pickling and loading with temporary file
    with tempfile.TemporaryFile() as f:
        pickle.dump(road_local, f)
        f.seek(0)
        road_loaded = pickle.load(f)
        plot_road(road_loaded)
