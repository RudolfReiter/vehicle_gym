import math
from abc import ABC
from copy import copy
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from enum import Enum, auto
from typing import List, Tuple

import numpy as np
from scipy import interpolate

from vehiclegym.utils.helpers import add_array2array, resample_array, FrozenClass


class VehicleObstacleModel(Enum):
    ELIPSE = auto()
    CIRCLES = auto()
    PANOS = auto()
    HYPERPLANE = auto()


def get_time_grid(n_nodes, time_disc, time_stretch_fac) -> np.ndarray:
    time_steps = get_time_steps(n_nodes, time_disc, time_stretch_fac)
    time_grid = np.hstack((np.array([0]), np.cumsum(time_steps)))
    return time_grid


def get_time_steps(n_nodes, time_disc, time_stretch_fac) -> np.ndarray:
    time_steps = np.ones((n_nodes - 1,)) * time_disc
    for i in range(time_steps.shape[0]):
        time_steps[i] = time_steps[i] * (1 + i * time_stretch_fac)
    return time_steps


@dataclass_json
@dataclass
class PlannerOptions(FrozenClass):
    n_nodes: int = 30  # Time steps for optimization horizon
    time_disc: float = 0.1
    max_iter: int = 5000
    v_max_terminal_set: float = 15
    w_acc_slack_l1: float = 1e7
    w_bound_slack_l1: float = 1e7
    w_obst_slack_l1: float = 0
    w_acc_slack_l2: float = 2e6
    w_bound_slack_l2: float = 1e7
    w_obst_slack_l2: float = 1e7
    debug_mode: bool = True
    increase_opp: float = 0.
    auto_size_circles: bool = False
    obstacle_model: VehicleObstacleModel = VehicleObstacleModel.ELIPSE
    use_lifting: bool = False
    deactivate_obstacle_avoidance: bool = False
    distance_obstacle_awareness: float = 0.
    use_cython: bool = False
    condensing_relative: float = 1  # relative value of condensing (1 means no condensing)
    qp_tolerance: float = 1e-6
    obstacle_increase_per_sec: float = 0.0
    time_stretch_fac: float = 0.00

    q0: List = None
    r0: List = None
    qn: List = None
    velocity_set: float = None
    integrator_stages: int = 1
    integrator_steps: int = 2


@dataclass
class PlanningData:
    """ Data for the animation. Each field should be of shape (n_states, n_trajectory_states, overall_iterations)"""
    x: np.ndarray = np.array([])
    u: np.ndarray = np.array([])
    x_opp_predicted: np.ndarray = np.array([])
    x_opp_predicted_all: List = field(default_factory=lambda: [])
    t: np.ndarray = np.array([])
    slacks2: np.ndarray = np.array([])
    slacks1: np.ndarray = np.array([])
    a_cost_ds: float = 0
    a_cost_dn: float = 0
    a_q_matrix: np.ndarray = np.ndarray([])


@dataclass
class PlanningDataContainer:
    x: np.ndarray = np.array([])
    u: np.ndarray = np.array([])
    x_opp_predicted: np.ndarray = np.array([])
    x_opp_predicted_all: List = field(default_factory=lambda: [])
    t: np.ndarray = np.array([])
    x_flat: np.ndarray = np.array([])
    u_flat: np.ndarray = np.array([])
    t_flat: np.ndarray = np.array([])
    slacks2: np.ndarray = np.array([])
    slacks1: np.ndarray = np.array([])
    n_trajectory_states_truncated: float = 0

    def add(self, planning_data: PlanningData):
        self.x = add_array2array(self.x, planning_data.x)
        self.slacks1 = add_array2array(self.slacks1, planning_data.slacks1, axis=1)
        self.u = add_array2array(self.u, planning_data.u)
        self.t = add_array2array(self.t, planning_data.t, axis=1)
        if planning_data.x_opp_predicted is not None:
            self.x_opp_predicted = add_array2array(self.x_opp_predicted, planning_data.x_opp_predicted)
        for i in range(len(planning_data.x_opp_predicted_all)):
            if len(self.x_opp_predicted_all) <= 0:
                for j in range(len(planning_data.x_opp_predicted_all)):
                    self.x_opp_predicted_all.append(np.array([]))
            if planning_data.x_opp_predicted_all[i] is not None:
                self.x_opp_predicted_all[i] = add_array2array(self.x_opp_predicted_all[i],
                                                              planning_data.x_opp_predicted_all[i])

    def flatten_data(self) -> bool:
        if self.t.shape[1] <= 1:
            self.t = self.t.transpose()
        t_skipped = self.t[0, 1] - self.t[0, 0]
        idx_skipped = np.argmin(np.abs(t_skipped - self.t[:, 0]))
        x_truncated = self.x[:, :idx_skipped, :]
        u_truncated = self.u[:, :idx_skipped, :]
        t_truncated = self.t[:idx_skipped, :]

        (n_states, self.n_trajectory_states_truncated, overall_iterations) = x_truncated.shape
        n_samples = self.n_trajectory_states_truncated * overall_iterations
        self.x_flat = x_truncated.transpose(0, 2, 1).reshape(n_states, n_samples)
        self.t_flat = t_truncated.transpose().reshape(n_samples)
        if self.u.shape[0] > 0:
            (n_controls, _, _) = self.u.shape
            self.u_flat = u_truncated.transpose(0, 2, 1).reshape(n_controls, n_samples)
        return True

    def resample(self, original_delta_t: float, output_delta_t: float) -> bool:
        self.n_trajectory_states_truncated = self.n_trajectory_states_truncated * original_delta_t / output_delta_t
        self.x_flat = resample_array(self.x_flat, self.t_flat, output_delta_t, axis=1)
        t_flat_orig = copy(self.t_flat)
        self.t_flat = resample_array(self.t_flat, self.t_flat, output_delta_t)
        if self.u.shape[0] > 0:
            self.u_flat = resample_array(self.u_flat, t_flat_orig, output_delta_t, axis=1)
        return True


class Planner(ABC):
    def __init__(self):
        self.sol = None
        self.x_full = None
        self.u_full = None
        self.slacks2_full = None
        self.slacks1_full = None
        self.x_opp_prediction = None
        self.opp_model = None
        self.time_pred = None
        self.nodes = None
        self.n_nodes = None
        self.time_grid = None
        self.current_slack = None

    def get_formatted_solution(self, t0: float = 0) -> PlanningData:
        data = PlanningData()
        data.x = copy(self.x_full)
        data.u = copy(self.u_full)
        data.x_opp_predicted = copy(self.x_opp_prediction)
        data.t = t0 + self.time_grid
        data.slacks1 = copy(self.current_slacks)
        if hasattr(self, "x_opposing_containers"):
            data.x_opp_predicted_all = copy(self.x_opposing_containers)
        return data

    def get_opposing_trajectory(self) -> np.ndarray:
        trajectory = []
        for cnt in range(self.n_nodes):
            trajectory.append(self.sol.value(self.x_opposing_container[cnt]))
        return np.array(trajectory).transpose()

    def get_state_at_time(self, relative_time: float) -> np.ndarray:
        assert relative_time <= self.time_pred
        assert relative_time >= 0
        div_rest = math.fmod(relative_time, self.time_disc)
        if abs(div_rest) < 1e-6 or abs(div_rest - self.time_disc) < 1e-6:
            index = int(relative_time / self.time_disc)
            ego_state = self.x_full[:, index]
        else:
            t = np.arange(0, self.time_pred + 1e-6, self.time_disc)
            fx = interpolate.interp1d(t, self.x_full)
            ego_state = fx(relative_time)
        return ego_state

    def get_states_until_time(self, end_time: float, sample_time: float = -0.1):
        if sample_time < 0:
            sample_time = self.time_disc  # Hackmack
        assert end_time <= self.time_pred
        assert end_time >= 0
        div_rest = math.fmod(end_time, sample_time)
        assert abs(div_rest) < 1e-6 or abs(div_rest - sample_time) < 1e-6

        if abs(sample_time - self.time_disc) < 1e-6:
            index = int(end_time / self.time_disc) + 1
            ego_states = self.x_full[:, 0:index]
        else:
            t = np.arange(0, self.time_pred, self.time_disc)
            fx = interpolate.interp1d(t, self.x_full)
            t_out = np.arange(0, end_time + sample_time, sample_time)
            ego_states = fx(t_out)
        return ego_states

    def get_prediction_until_time(self, end_time: float, sample_time: float = -0.1):
        if sample_time < 0:
            sample_time = self.time_disc  # Hackmack
        assert end_time <= self.time_pred
        assert end_time >= 0
        div_rest = math.fmod(end_time, sample_time)
        assert abs(div_rest) < 1e-6 or abs(div_rest - sample_time) < 1e-6

        if abs(sample_time - self.time_disc) < 1e-6:
            index = int(end_time / self.time_disc) + 1
            pred_states = self.x_opp_prediction[:, 0:index]
        else:
            t = np.arange(0, self.time_pred, self.time_disc)
            fx = interpolate.interp1d(t, self.x_opp_prediction)
            t_out = np.arange(0, end_time + sample_time, sample_time)
            pred_states = fx(t_out)
        return pred_states

    def get_controls_until_time(self, end_time: float, sample_time: float = -0.1):
        if sample_time < 0:
            sample_time = self.time_disc  # Hackmack
        assert end_time <= self.time_pred
        assert end_time >= 0
        div_rest = math.fmod(end_time, sample_time)
        assert abs(div_rest) < 1e-6 or abs(div_rest - sample_time) < 1e-6

        if abs(sample_time - self.time_disc) < 1e-6:
            index = int(end_time / self.time_disc) + 1
            ego_controls = self.u_full[:, 0:index]
        else:
            t = np.arange(0, self.time_pred, self.time_disc)
            fx = interpolate.interp1d(t, self.u_full)
            t_out = np.arange(0, end_time + sample_time, sample_time)
            ego_controls = fx(t_out)
        return ego_controls
