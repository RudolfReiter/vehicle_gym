from typing import List, TYPE_CHECKING, Tuple
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline
import casadi as cs
from scipy.signal import savgol_filter


class FrozenClass(object):
    __isfrozen = False

    def __setattr__(self, key, value):
        if self.__isfrozen and not hasattr(self, key):
            raise TypeError("%r is a frozen class" % self)
        object.__setattr__(self, key, value)

    def _freeze(self):
        self.__isfrozen = True


def integrate_curvature(
    s_grid: np.ndarray, kappa_grid: np.ndarray, start_point_cartesian: np.ndarray = None
):
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


#
# def compute_curvature(x: np.ndarray, y: np.ndarray, smoothing=3) -> Tuple[np.ndarray, np.ndarray]:
#     # compute s
#     dx = np.diff(x)
#     dy = np.diff(y)
#     s = np.append(0, np.sqrt(dx ** 2 + dy ** 2))
#     s = np.cumsum(s)
#
#     dxds = UnivariateSpline(s, x, k=3, s=smoothing).derivative(1)(s)
#     d2dxds = UnivariateSpline(s, x, k=3, s=smoothing).derivative(2)(s)
#     dyds = UnivariateSpline(s, y, k=3, s=smoothing).derivative(1)(s)
#     d2dyds = UnivariateSpline(s, y, k=3, s=smoothing).derivative(2)(s)
#
#     denominator = pow(dxds * dxds + dyds * dyds, 1.5)
#     kappa = ((dxds * d2dyds - dyds * d2dxds) / denominator)
#
#     return s, kappa
