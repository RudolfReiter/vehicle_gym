import itertools
import json
import os
import sys
import warnings
from typing import List, TYPE_CHECKING, Tuple
import casadi as cs
import matplotlib
import numpy as np
import pandas as pd
import scipy
from scipy.interpolate import UnivariateSpline
from vehicle_models.model_kinematic import KinematicModelParameters
from vehicle_models.model_kinematic_frenet import FrenetModel
from vehiclegym.utils.automotive_datastructures import FrenetTrajectory
from pathlib import Path
from data import DATAPATH

if TYPE_CHECKING:
    from vehiclegym.road import Road, RoadOptions


def powspace(start, stop, power, num):
    start = np.power(start, 1 / float(power))
    stop = np.power(stop, 1 / float(power))
    return np.power(np.linspace(start, stop, num=num), power)


def get_extended_diag(vec: np.ndarray, n: int):
    n_vec = vec.shape[0]
    diag_vec = np.zeros((n,))
    diag_vec[0:n_vec] = vec
    return np.diag(diag_vec)


def extend_by_zeros(vec: np.ndarray, n_set: int):
    n_vec = vec.shape[0]
    out_vec = np.zeros((n_set,))
    out_vec[0:n_vec] = vec
    return out_vec


def extend_by_dual(state_f: np.ndarray, road: "Road"):
    n_vec = state_f.shape[0]
    out_vec = np.zeros((n_vec + 3,))
    out_vec[0:n_vec] = state_f
    state_c = road.transform_trajectory_f2c(trajectory_f=FrenetTrajectory(state_f))
    out_vec[-3] = state_c.x
    out_vec[-2] = state_c.y
    out_vec[-1] = state_c.phi
    return out_vec


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class FrozenClass(object):
    __isfrozen = False

    def __setattr__(self, key, value):
        if self.__isfrozen and not hasattr(self, key):
            raise TypeError("%r is a frozen class" % self)
        object.__setattr__(self, key, value)

    def _freeze(self):
        self.__isfrozen = True


def latexify(fontsize: int = 10):
    params = {'backend': 'ps',
              'text.latex.preamble': r"\usepackage{gensymb} \usepackage{amsmath}",
              'axes.labelsize': fontsize,
              'axes.titlesize': fontsize,
              'legend.fontsize': fontsize,
              'xtick.labelsize': fontsize,
              'ytick.labelsize': fontsize,
              'text.usetex': True,
              'font.family': 'serif'
              }

    matplotlib.rcParams.update(params)


def latexify_debug():
    params = {'backend': 'ps',
              'text.latex.preamble': r"\usepackage{gensymb} \usepackage{amsmath}",
              'axes.labelsize': 20,
              'axes.titlesize': 20,
              'legend.fontsize': 20,
              'xtick.labelsize': 20,
              'ytick.labelsize': 20,
              'text.usetex': True,
              'font.family': 'serif'
              }

    matplotlib.rcParams.update(params)


def compute_velocity_upper_bound(road_options: "RoadOptions", vehicle_parameters: KinematicModelParameters,
                                 weight_delta: float) -> [float, float]:
    """
    Function computes a set of safe states related to a maximum curvature, a vehicle model and alpha
    (the vehicle heading error in frenet coordinates), and delta (steering angle) fixed to 0
    So the safe set is:
    S= {x |
        n \in [-nmax, nmax],
        v < v_max,
        alpha = 0,
        delta = 0}

    :param road_options: total width of road and maximum curvature of the road appearing on the track
                        (used for worst case estimation). Either road_width or maximum kappa must be set.
    :param vehicle_parameters: Model parameter struct
    :param weight_velocity: a weight, how much more the velocity should be increased as opposed to the distance of the
                            bounds should be lowered. Both need to be traded off.
    :return: v_max, n_max
    """
    if road_options.road_width is None and road_options.random_road_parameters.maximum_kappa is None:
        warnings.warn("Wrong usage of final set computation function. Returned zero set")
        return 0, 0

    road_width = road_options.road_width
    if road_options.random_road_parameters.maximum_kappa is not None:
        maximum_curvature = road_options.random_road_parameters.maximum_kappa
    else:
        maximum_curvature = 1 / (road_options.road_width / 2)

    s_grid = np.linspace(0, 100, 100)
    kappa_grid = np.ones_like(s_grid) * maximum_curvature
    road_options = RoadOptions(s_grid=s_grid, kappa_grid=kappa_grid, road_width=road_width)
    road = Road(road_options)
    opti = cs.Opti()
    n_nodes = 10
    X = opti.variable(5, n_nodes)
    U = opti.variable(2, n_nodes - 1)
    v0 = opti.variable(1)
    delta_bound = opti.variable(1)
    p_kappa = opti.parameter(road.s_grid_.shape[0])

    model = FrenetModel(s_grid=s_grid,
                        p_kappa=p_kappa,
                        params=vehicle_parameters)
    options = {'print_time': 0,
               'ipopt': {'max_iter': 5000,
                         'print_level': 0,
                         'acceptable_tol': 1e-7,
                         'acceptable_obj_change_tol': 1e-7}}
    opti.solver("ipopt", options)

    t_end = vehicle_parameters.mass * v0 / vehicle_parameters.maximum_deceleration_force
    t_disc = 1 / (n_nodes - 1) * t_end

    k1 = model.f_ode(model.x_states, model.u_controls)
    k2 = model.f_ode(model.x_states + t_disc / 2 * k1, model.u_controls)
    k3 = model.f_ode(model.x_states + t_disc / 2 * k2, model.u_controls)
    k4 = model.f_ode(model.x_states + t_disc * k3, model.u_controls)
    x_next = model.x_states + t_disc / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    f_integrator = cs.Function('F_int', [model.x_states, model.u_controls], [x_next])

    for k in range(n_nodes - 1):
        opti.subject_to(X[:, k + 1] == f_integrator(X[:, k], U[:, k]))  # close the gaps

    opti.subject_to(U[0, :] == -vehicle_parameters.maximum_deceleration_force)
    n0 = -road_width / 2 + vehicle_parameters.safety_radius
    opti.subject_to(X[0, 0] == 0)
    opti.subject_to(X[1, 0] == n0)
    opti.subject_to(X[2, 0] == 0)
    opti.subject_to(X[3, 0] == v0)
    opti.subject_to(X[4, 0] == delta_bound)

    opti.subject_to(v0 >= 0)

    opti.subject_to(-(road.road_options_.road_width / 2 - vehicle_parameters.safety_radius) <=
                    (X[1, :] <= road.road_options_.road_width / 2 - vehicle_parameters.safety_radius))
    opti.subject_to(
        - vehicle_parameters.maximum_steering_angle <= (X[4, :] <= vehicle_parameters.maximum_steering_angle))
    opti.subject_to(-vehicle_parameters.maximum_alpha <= (X[2, :] <= vehicle_parameters.maximum_alpha))
    opti.subject_to(
        - vehicle_parameters.maximum_steering_rate <= (U[1, :] <= vehicle_parameters.maximum_steering_rate))

    for k in range(n_nodes):
        lat_acc = model.f_a_lat(X[:, k])
        opti.subject_to(-vehicle_parameters.maximum_lateral_acc <=
                        (lat_acc <= vehicle_parameters.maximum_lateral_acc))
    opti.minimize(-v0 - delta_bound * weight_delta)

    opti.set_value(p_kappa, kappa_grid)
    opti.set_initial(v0, 1)
    res = opti.solve()
    return res.value(v0), res.value(delta_bound)


def compute_curvature(p_xy: np.ndarray):
    """
    :param p_xy: array of size (n,2) representing Cartesian 2D points
    :return: curvature and path length
    """
    assert p_xy.shape[1] == 2
    # first derivatives
    dx = np.gradient(p_xy[:, 0])
    dy = np.gradient(p_xy[:, 1])

    # second derivatives
    d2x = np.gradient(dx)
    d2y = np.gradient(dy)

    # calculation of curvature from the typical formula
    curvature = (dx * d2y - d2x * dy) / (dx * dx + dy * dy) ** 1.5
    path_length = np.cumsum(np.sqrt(dx ** 2 + dy ** 2))

    return curvature, path_length


def dataclass2json(dataclass, filename: str, relpath: str):
    """
    This function writs a readable json file with the data stored in the data class
    :param dataclass: specifice dataclass which should be saved. Not allowed to contain np.arrays
    :param filename: filename should be related to the data class
    :param relpath: relative path to the data/parameter folder
    """
    # get file name and path
    # path = root_path(ignore_cwd=False) + os.sep + 'data' + os.sep + 'parameter' + os.sep + relpath
    path = DATAPATH + os.sep + 'parameter' + os.sep + relpath
    filename_abs = path + os.sep + filename

    # serialize data
    data = dataclass.to_json()
    data = json.loads(data)

    # save data
    with open(filename_abs, 'w') as f:
        json.dump(data, f, indent=4)


def get_tracking_state(state_traj: np.ndarray,
                       state_meas: np.ndarray,
                       diff_state_max: np.ndarray,
                       par: Tuple = (-0.5, 5)):
    """
    Get a partial mix of two states, based on how far their difference is away of the threshold value. If
    close to the threshold, take measured state. If difference is close to zero, take trajectory state.
    :param state_traj: Trajectory state (predicted but not measured)
    :param state_meas: Current measured state
    :param diff_state_max: Threshold maximum of state difference
    :param par: parameter for sigmoid function. (center, steepness)
    :return: mix of the two states.
    """
    print(state_traj - state_meas)
    treshold_measure = (np.max(np.abs(state_traj - state_meas) / diff_state_max) - 1)
    mu_partial_measure = 1 / (1 + np.exp(-(treshold_measure - par[0]) * par[1]))
    mu_partial_traj = 1 - mu_partial_measure
    states_forward = mu_partial_measure * state_meas + mu_partial_traj * state_traj
    print(mu_partial_traj)
    return states_forward


def read_track(name: str = "Spielberg", oversample_factor: int = 2):
    df = pd.read_csv(DATAPATH + os.sep + "racetracks/" + name + ".csv")
    x = np.expand_dims(np.array(df.get("# x_m").array), 1)
    N = x.shape[0]
    N_up = N * oversample_factor
    x = scipy.signal.resample(x, N_up)
    y = np.expand_dims(np.array(df.get("y_m").array), 1)
    y = scipy.signal.resample(y, N_up)
    p_xy = np.hstack((x, y))
    nr = np.array(df.get("w_tr_right_m").array)
    nr = scipy.signal.resample(nr, N_up)
    nl = np.array(df.get("w_tr_left_m").array)
    nl = scipy.signal.resample(nl, N_up)
    return p_xy, nl, nr


def json2dataclass(dataclass_type, filename: str, relpath: str):
    """
    Reads a json file that stores data of a certain dataclass and creates an instance with according values
    :param dataclass_type: dataclass (not an instance)
    :param filename: filename of json
    :param relpath: relative path to data/parameters
    :return: an instance of the dataclass with according values
    """
    # get file name and path
    path = DATAPATH + os.sep + 'parameter' + os.sep + relpath
    filename_abs = path  + filename

    # Opening JSON file
    with open(filename_abs) as f:
        data = json.load(f)
    dataclass_ = dataclass_type.from_dict(data)
    return dataclass_


def resample_array(input_array: np.ndarray, original_t: np.ndarray, output_delta_t: float, kind: str = 'linear',
                   axis=0):
    t_output = np.linspace(original_t[0], original_t[-1],
                           int(np.round((original_t[-1] - original_t[0]) / output_delta_t + 1)))
    interp_fun = scipy.interpolate.interp1d(original_t, input_array, kind=kind, axis=axis)
    return interp_fun(t_output)


def resample_actions(actions: List, number_of_simulated_steps: int, original_delta_t, output_delta_t):
    action0 = list(itertools.chain.from_iterable(itertools.repeat(x, number_of_simulated_steps) for x in actions[0]))
    action1 = list(itertools.chain.from_iterable(itertools.repeat(x, number_of_simulated_steps) for x in actions[1]))
    action0 = np.array(action0)
    action1 = np.array(action1)
    actions = np.vstack((action0, action1))
    actions = resample_array(actions, original_delta_t, output_delta_t, kind='zero')
    return actions


def add_array2array(array_base: np.ndarray, array_add: np.ndarray, axis: int = 2) -> np.ndarray:
    """ Adding a value to an array by expansion. If the array is not initialized it should not be expanded."""
    # assert (array_base.shape[axis] == len(array_add))
    if array_base.size == 0:
        array_base = np.expand_dims(array_add, axis=axis)
    else:
        array_add_transformed = np.expand_dims(array_add, axis=axis)
        array_base = np.append(array_base, array_add_transformed, axis=axis)
    return array_base


# const double denominator = pow(dxds * dxds + dyds * dyds, 1.5);
#        kappa_[i] = (fabs(denominator) > eps) ? ((dxds * d2dyds - dyds * d2dxds) / denominator) : 0;

if __name__ == "__main__":
    t = np.linspace(0, np.pi, 10)
    R = 1
    x_test = R * np.cos(t)
    y_test = R * np.sin(t)
    s_test, kappa_test = compute_curvature(x_test, y_test)
    print(s_test)
    print(kappa_test)
    print(kappa_test.shape)
    print(s_test.shape)
    print(x_test.shape)
