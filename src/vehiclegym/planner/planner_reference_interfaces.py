from copy import copy
from dataclasses import dataclass
import numpy as np
from typing import List
from abc import ABC, abstractmethod
from dataclasses_json import dataclass_json
import casadi as cs


@dataclass
class ReferenceValues:
    Q: np.ndarray
    Q_n: np.ndarray
    R: np.ndarray
    x_ref: np.ndarray
    u_ref: np.ndarray


@dataclass_json
@dataclass
class CostInterfaceOptions:
    default_action: List = None
    maximum_action: List = None
    minimum_action: List = None
    q0: List = None
    r0: List = None
    qn: List = None
    time_grid: List = None
    check_sizes: bool = False


def check_ref_matrix_sizes(ref_values: ReferenceValues,
                           ref_options: CostInterfaceOptions) -> bool:
    valid = True
    shape_Q = ref_values.Q.shape
    shape_Q_n = ref_values.Q_n.shape
    shape_R = ref_values.R.shape
    shape_x_ref = ref_values.x_ref.shape
    shape_u_ref = ref_values.u_ref.shape
    n_nodes = ref_options.time_grid.__len__()
    if len(shape_Q) != 3 or len(shape_Q_n) != 2 or \
            len(shape_R) != 3 or len(shape_x_ref) != 2 or \
            len(shape_u_ref) != 2:
        valid = False
    if shape_Q[0] != n_nodes - 1 or \
            shape_u_ref[0] != n_nodes - 1 or \
            shape_R[0] != n_nodes - 1 or \
            shape_x_ref[0] != n_nodes or \
            shape_u_ref[0] != n_nodes - 1:
        valid = False
    return valid


def repeat_memoryless(arr, count):
    res = np.broadcast_to(arr, (count,) + arr.shape)
    return res


def repeat(arr, count):
    return np.stack([arr for _ in range(count)], axis=0)


class CostInterface(ABC):
    def __init__(self, options: CostInterfaceOptions):
        self.options = options
        self.Q0 = np.diag(options.q0)
        self.R0 = np.diag(options.r0)
        self.Qn = np.diag(options.qn)

    def forward(self, x0: np.ndarray, action: np.ndarray = None, t_current: float = None) -> ReferenceValues:
        if action is None:
            action = self.options.default_action
        reference_values = self.forward_(action=action, x0=x0, t_current=t_current)
        if self.options.check_sizes:
            valid = check_ref_matrix_sizes(reference_values, self.options)
            if not valid:
                Exception()
        return reference_values

    @abstractmethod
    def forward_(self, action: np.ndarray, x0: np.ndarray, t_current: float = None) -> ReferenceValues:
        pass

    @abstractmethod
    def set_lateral_reference(self, n: float = None):
        pass

    def get_max_action(self) -> np.ndarray:
        return np.array(self.options.maximum_action)

    def get_min_action(self) -> np.ndarray:
        return np.array(self.options.minimum_action)


class DiffableReferenceInterface(CostInterface, ABC):
    def __init__(self, options: CostInterfaceOptions):
        super().__init__(options)

    @abstractmethod
    def get_jacobian(self, par_sensitivities: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def get_par_symbolic(self):
        pass


@dataclass_json
@dataclass
class InterfaceFrenet3Options(CostInterfaceOptions):
    gamma: float = 1  # decay per second
    ignore_terminal_value: bool = False


class InterfaceFrenet3(CostInterface):
    def __init__(self, options: InterfaceFrenet3Options):
        super().__init__(options)
        self.time_grid = None
        self.n_nodes = 0
        self.nx = self.Q0.shape[0]
        self.nu = self.R0.shape[0]
        self.ny = self.nx + self.nu
        self.gamma_vec = self.options.gamma**np.arange(self.n_nodes-1)

    def set_time_grid(self, time_grid: np.ndarray):
        self.time_grid = time_grid
        self.n_nodes = len(self.time_grid)
        self.options.time_grid = time_grid

        # setting gamma such that
        exponent_vec = np.diff(self.time_grid) * np.arange(self.n_nodes - 1) / np.diff(self.time_grid)[0]
        self.gamma_vec = self.options.gamma ** exponent_vec

    def set_lateral_reference(self, n: float = None):
        self.options.default_action[0] = n

    def get_stage_cost(self, x0: np.ndarray, x_next: np.ndarray, u: np.ndarray, action=None):
        dt = self.time_grid[1]
        if x0 is None:
            x0 = copy(x_next)
        if len(x0.shape) == 2:
            x0 = x0[:, 0]
        cost = 0
        cost += u.T @ self.R0 @ u
        if action is None:
            action = self.options.default_action
        [n_set, v_set, n_weight] = action

        Q = copy(self.Q0)
        Q[1, 1] = 10 ** n_weight
        s_ref = x0[0] + dt * v_set
        x_ref = np.array([s_ref, n_set, 0, v_set, 0])
        delta_x = x_next - x_ref
        cost += delta_x.T @ Q @ delta_x * self.gamma_vec[1]
        return cost * dt

    def forward_(self, action: np.ndarray, x0: np.ndarray, t_current: float = None) -> ReferenceValues:
        [n_set, v_set, n_weight] = [action[0], action[1], action[2]]
        assert all(np.array([n_set, v_set, n_weight]) <= self.options.maximum_action)
        assert all(np.array([n_set, v_set, n_weight]) >= self.options.minimum_action)

        # Set weights
        Q = copy(self.Q0)
        Q[1, 1] = 10 ** n_weight
        if self.options.gamma >= 1-1e-9:
            Q = repeat_memoryless(Q, self.n_nodes - 1)
            R = repeat_memoryless(self.R0, self.n_nodes - 1)
        else:
            Q = np.tile(np.expand_dims(Q,axis=0),(self.n_nodes - 1,1,1))
            Q = np.expand_dims(np.expand_dims(self.gamma_vec,-1),-1)*Q
            R = np.tile(np.expand_dims(self.R0, axis=0), (self.n_nodes - 1,1, 1))
            R = np.expand_dims(np.expand_dims(self.gamma_vec,-1),-1)*R

        if self.options.ignore_terminal_value:
            self.Qn = Q[-1,...]*(self.time_grid[-1]-self.time_grid[-2])
            # todo: change to the above but it does not work
            #self.Qn = Q[-1, ...]


        # Set reference trajectory related to set speed
        x_ref = np.zeros((self.n_nodes, self.nx))
        for stage in range(self.n_nodes - 1):
            s_ref = x0[0] + self.time_grid[stage] * v_set
            x_ref[stage, :5] = np.array([s_ref, n_set, 0, v_set, 0])

        s_ref = x0[0] + self.time_grid[-1] * v_set
        x_ref[-1, :5] = np.array([s_ref, n_set, 0, v_set, 0])

        u_ref = np.zeros((self.n_nodes - 1, self.nu))

        reference_values = ReferenceValues(Q=Q, R=R, Q_n=self.Qn, x_ref=x_ref, u_ref=u_ref)
        return reference_values


@dataclass_json
@dataclass
class InterfaceLaneChangeOptions(CostInterfaceOptions):
    weight_lane_sticking: float = 1e2
    weight_switch_s: float = 1e5
    weight_switch_n: float = 1e4


class InterfaceLaneChange(CostInterface):
    def __init__(self, options: InterfaceLaneChangeOptions):
        super().__init__(options)
        self.time_grid = np.array(options.time_grid)
        self.n_nodes = len(options.time_grid)
        self.nx = self.Q0.shape[0]
        self.nu = self.R0.shape[0]
        self.ny = self.nx + self.nu

    def forward_(self, action: np.ndarray, x0: np.ndarray, t_current: float = None) -> ReferenceValues:
        [n_lane_1, n_lane_2, n_switch, s_switch, t_switch, v_ref] = [*action]
        assert all(np.array(action) <= self.options.maximum_action)
        assert all(np.array(action) >= self.options.minimum_action)

        # get lane switch node
        if t_current is not None:
            t_switch -= t_current
        idx_switch = np.argmin(np.abs(t_switch - self.time_grid))
        t_diff_switch = self.time_grid - t_switch

        do_lane_change = True
        if idx_switch == 0:
            do_lane_change = False

        # Set weights
        Q = copy(self.Q0)
        Q = repeat(Q, self.n_nodes - 1)

        # set standard R for all stages
        R = repeat_memoryless(self.R0, self.n_nodes - 1)
        u_ref = np.zeros((self.n_nodes - 1, self.nu))

        x_ref = np.zeros((self.n_nodes, self.nx))

        if do_lane_change:

            # set high weight on lane change point
            # Q[idx_switch, 0, 0] = self.options.weight_switch_s
            # Q[idx_switch, 1, 1] = self.options.weight_switch_n

            fac_weight = 1
            duration_lanechange = 1
            fac_lane_change = 20 / duration_lanechange
            fac_pos_s = 1
            Q[:, 0, 0] = self.options.weight_switch_s * np.exp(-fac_weight * t_diff_switch[:-1] ** 2) + self.Q0[0, 0]
            Q[:, 1, 1] = self.options.weight_switch_n * np.exp(-0.1 * t_diff_switch[:-1] ** 2) + self.Q0[1, 1]

            n_ref = n_lane_1 + 1 / (1 + np.exp(-(fac_lane_change * t_diff_switch))) * (n_lane_2 - n_lane_1)

            v_ref_s = (s_switch - x0[0]) / t_switch
            s_ref_pre = x0[0] + self.time_grid * v_ref_s
            s_ref_post = np.maximum(s_switch + t_diff_switch * v_ref, x0[0])
            s_ref = s_ref_pre + 1 / (1 + np.exp(-(fac_pos_s * t_diff_switch))) * (s_ref_post - s_ref_pre)
            # print(s_ref)
            # from matplotlib import pyplot as plt
            # plt.plot(s_ref_pre)
            # plt.plot(s_ref_post)
            # plt.plot(s_ref)
            # plt.show()

            # Set reference trajectory related to set speed
            for stage in range(self.n_nodes - 1):
                if stage <= idx_switch:
                    v_ref = v_ref_s
                x_ref[stage] = np.array([s_ref[stage], n_ref[stage], 0, v_ref, 0])

            x_ref[-1] = np.array([s_ref[-1], n_ref[-1], 0, v_ref, 0])
        else:
            for stage in range(self.n_nodes - 1):
                n_ref = n_lane_2
                s_ref = x0[0] + self.time_grid[stage] * v_ref
                x_ref[stage] = np.array([s_ref, n_ref, 0, v_ref, 0])
            s_ref = x0[0] + self.time_grid[-1] * v_ref
            x_ref[-1] = np.array([s_ref, n_lane_2, 0, v_ref, 0])

        reference_values = ReferenceValues(Q=Q, R=R, Q_n=self.Qn, x_ref=x_ref, u_ref=u_ref)
        return reference_values
