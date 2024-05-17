import numpy as np
import casadi as cs
import matplotlib.pyplot as plt

from typing import Tuple
from scipy.signal import savgol_filter

def integrate_curvature(
    s_grid: np.ndarray, kappa_grid: np.ndarray, start_point_cartesian: np.ndarray = None
):
    """
    Integrate the curvature, to compute Cartesian coordinates x and y.
    param s_grid: array of path length
    param kappa_grid: array of curvature points related to the path length s, i.e., kappa(s)
    start_point_cartesian: optional starting state of the curve (x,y,phi,v,delta)
    """
    # Start CASADI definition to integrate kappa(s) to get x, y and phi of the curve
    interp_s2kappa = cs.interpolant("interp_s2kappa", "bspline", [s_grid], kappa_grid)

    s = cs.MX.sym("s")
    x = cs.MX.sym("x")
    y = cs.MX.sym("y")
    phi = cs.MX.sym("phi")

    states = cs.vertcat(x, y, phi)

    rhs = cs.vertcat(cs.cos(phi), cs.sin(phi), interp_s2kappa(s))

    dae = {"x": states, "t": s, "ode": rhs}

    opts = {}
    opts["reltol"] = 1e-15
    opts["abstol"] = 1e-15
    opts["fsens_err_con"] = True
    opts["t0"] = 0
    opts["grid"] = s_grid
    opts["output_t0"] = True

    integrator = cs.integrator("integrator", "cvodes", dae, opts)
    if start_point_cartesian is None:
        x_c = np.array([0, 0, 0, 0, 0])
    else:
        x_c = start_point_cartesian
    sol = integrator(x0=[x_c[0], x_c[1], x_c[2]])

    x_ = np.array(sol["xf"])[0, :]
    y_ = np.array(sol["xf"])[1, :]
    phi_ = np.array(sol["xf"])[2, :]
    return x_, y_, phi_


def compute_curvature(
    p_xy: np.ndarray, filter_coefs: Tuple = (51, 5), do_plot: bool = False
):
    """
    Given points px and py, compute the curvature, path length and heading angle

    :param p_xy: array of size (n,2) representing Cartesian 2D points
    :param filter_coefs: Savgol filter coefs. used to smoothen the curve
    :param do_plot: Plot curvature and smoothed curvature
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
    if len(curvature)>6:
        curvature[0] = curvature[2]
        curvature[1] = curvature[2]
        curvature[-1] = curvature[-3]
        curvature[-2] = curvature[-3]
    curvature2 = savgol_filter(curvature, filter_coefs[0], filter_coefs[1])
    # curvature2 = np.clip(curvature2,-0.005, 0.005)

    # curvature = curvature2
    path_s = np.cumsum(np.sqrt(dx**2 + dy**2))
    path_length = path_s - path_s[0]
    phi = np.arctan2(dy, dx)

    if do_plot:
        plt.plot(path_length,curvature, label="original curvature")
        plt.plot(path_length,curvature2, label="smooth curvature")
        plt.xlabel(r"path length (m)")
        plt.ylabel(r"curvature ($\frac{\mathrm{1}}{\mathrm{m}}$)")
        plt.legend()
        plt.show()

    return curvature2, path_length, phi