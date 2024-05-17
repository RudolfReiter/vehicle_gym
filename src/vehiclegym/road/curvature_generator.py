import numpy as np
from scipy.signal import savgol_filter
from typing import Tuple
from dataclasses import dataclass
from vehiclegym.utils.helpers import FrozenClass


@dataclass
class RandomRoadParameters(FrozenClass):
    number_points: int = 100  # number of points
    delta_s: float = 2.0  # spacing of points
    mu: float = 0  # mean of random curvature
    sigma: float = 0.2,  # std of random curvature
    n_filter_conv: int = 10  # filter coefficient to smooth road
    n_filter_savgol: Tuple[int, int] = (31, 2)  # savgol filter coeffitients to smooth road
    maximum_kappa: float = None  # a maximum kappa, that should be related to the road curvature
    seed: int = -1  # randomization seed
    boundary_mu: float = 5  # mean value of boundary distance from centerline (equal for both sides)
    boundary_std: float = 0  # std deviation of boundary distance from centerline
    n_filter_savgol_boundary: Tuple[int, int] = (31, 2)  # savgol coefficients for random boundary smoothing


def generate_boundary(params: RandomRoadParameters = RandomRoadParameters()) -> [np.ndarray, np.ndarray]:
    if params.seed >= 0:
        np.random.seed(params.seed)
    boundary_left = np.random.normal(params.boundary_mu, params.boundary_std, params.number_points)
    boundary_right = np.random.normal(params.boundary_mu, params.boundary_std, params.number_points)
    boundary_left = savgol_filter(boundary_left, params.n_filter_savgol_boundary[0],
                                  params.n_filter_savgol_boundary[1])
    boundary_right = savgol_filter(boundary_right, params.n_filter_savgol_boundary[0],
                                   params.n_filter_savgol_boundary[1])
    return boundary_left, boundary_right


def generate_circle(number_points: int = 1000, delta_s: float = 0.1, radius: float = 10) -> [np.ndarray,
                                                                                             np.ndarray]:
    s_grid_ = np.linspace(0, (number_points - 1) * delta_s, number_points)
    kappa_grid_ = np.ones_like(s_grid_) / radius
    return s_grid_, kappa_grid_


def generate_random(params: RandomRoadParameters = RandomRoadParameters()) -> [np.ndarray, np.ndarray]:
    if params.seed >= 0:
        np.random.seed(params.seed)
    s_grid_ = np.linspace(0, (params.number_points - 1) * params.delta_s, params.number_points)
    kappa_grid_ = np.random.normal(params.mu, params.sigma, len(s_grid_))
    if params.maximum_kappa is not None:
        kappa_grid_ = np.clip(kappa_grid_, -params.maximum_kappa, params.maximum_kappa)

    kappa_grid_ = np.convolve(kappa_grid_, np.ones(params.n_filter_conv) / params.n_filter_conv, mode='same')
    kappa_grid_ = savgol_filter(kappa_grid_, params.n_filter_savgol[0], params.n_filter_savgol[1])
    return s_grid_, kappa_grid_
