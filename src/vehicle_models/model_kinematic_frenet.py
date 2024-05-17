from typing import List


from vehicle_models.utils import get_circle_rect_cover, halfspace, get_vertices_rectangle
import casadi as cs
import numpy as np
from acados_template import AcadosModel
from vehicle_models.model_kinematic import (KinematicVehicleModel,
                                            KinematicModelParameters)
from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from vehiclegym.utils.acados_param_handler import ParamHandler


class KinematicVehicleModelFrenet(KinematicVehicleModel):
    """ Base class for kinematic vehicle models with 5 states """

    def __init__(self, s_grid: np.ndarray,
                 param_handler: "ParamHandler",
                 params: KinematicModelParameters,
                 add_states: int = 0,
                 p_nr_np: np.ndarray = None,
                 p_nl_np: np.ndarray = None,
                 p_kappa_np: np.ndarray = None):
        super().__init__(params, param_handler=param_handler, add_states=add_states)
        self.name = "frenetModel"
        self.params_ = params

        self.n_grid = s_grid.shape[0]
        self.s_grid = s_grid

        self.state_s = self.x_states[0]
        self.state_n = self.x_states[1]
        self.state_alpha = self.x_states[2]

        self.interp_s2nl = cs.interpolant("interp_s2nl", "linear", [s_grid])
        self.interp_s2nr = cs.interpolant("interp_s2nr", "linear", [s_grid])

        if p_nr_np is None and param_handler is not None:
            self.p_kappa = param_handler.get_casadi("p_kappa")
            self.p_nl = param_handler.get_casadi("p_nl")
            self.p_nr = param_handler.get_casadi("p_nr")
            constr_left = self.interp_s2nl(self.state_s,
                                           self.p_nl) - self.state_n - params.safety_radius_boundary  # geq 0
            constr_right = self.interp_s2nr(self.state_s,
                                            self.p_nr) + self.state_n - params.safety_radius_boundary  # geq 0
            self.boundary_constr = [constr_left, constr_right]
        else:
            self.boundary_constr = [1]

        if p_kappa_np is not None:
            # the parameter kappa is non symbolic
            interp_s2kappa_tmp = cs.interpolant("interp_s2kappa", "linear", [s_grid], p_kappa_np)
            self.interp_s2kappa = lambda s, p: interp_s2kappa_tmp(s)  # we need the standard interface of 2 parameters
            self.a_lat_center_lane = -self.state_v ** 2 * self.interp_s2kappa(self.state_s, None)
        else:  # must be casadi type
            # the parameter kappa is symbolic an can be used as paramter in optimization problem
            self.interp_s2kappa = cs.interpolant("interp_s2kappa", "linear", [s_grid])
            self.a_lat_center_lane = -self.state_v ** 2 * self.interp_s2kappa(self.state_s, self.p_kappa)

        self.f_a_lat_center_lane = cs.Function('f_a_lat_center_lane', [self.x_states], [self.a_lat_center_lane],
                                               {"allow_free": True})


class SimplePredictiveModel(KinematicVehicleModelFrenet):
    """ Simple vehicle model to predict an opposing vehicles motion, which assumes constant velocity """

    def __init__(self, s_grid: np.ndarray, p_kappa, p_nl, p_nr, params: KinematicModelParameters):
        super().__init__(s_grid, p_kappa, p_nl, p_nr, params)
        # dx/dt = f(x,u)
        s_dot = self.state_v * cs.cos(self.state_alpha)
        self.rhs = [self.state_v * cs.cos(self.state_alpha),
                    self.state_v * cs.sin(self.state_alpha),
                    0,
                    0,
                    0]
        self.post_init()


class FrenetModel(KinematicVehicleModelFrenet):
    """ Basic most standard kinematic vehicle model """

    def __init__(self, s_grid: np.ndarray,
                 param_handler: "ParamHandler",
                 params: KinematicModelParameters,
                 p_kappa_np: np.ndarray = None, p_nl_np: np.ndarray = None, p_nr_np: np.ndarray = None
                 ):
        super().__init__(s_grid, param_handler=param_handler, params=params,
                         p_kappa_np=p_kappa_np, p_nl_np=p_nl_np, p_nr_np=p_nr_np)
        p_kappa = param_handler.get_casadi("p_kappa") if param_handler is not None else p_kappa_np
        length = params.length_front + params.length_rear
        s_dot = self.state_v * cs.cos(self.state_alpha) / (
                1 - self.state_n * self.interp_s2kappa(self.state_s, p_kappa))
        self.rhs[0] = s_dot
        self.rhs[1] = self.state_v * cs.sin(self.state_alpha)
        self.rhs[2] = self.state_v / length * cs.tan(self.state_delta) - s_dot * self.interp_s2kappa(
            self.state_s, p_kappa)

        self.post_init()


class FrenetModelWithObstaclesPnorm(FrenetModel):
    """ Kinematic vehicle model with obstacles represented as p-norm
    The distance measure for the ego vehicle is a cicular distance, the opposing vehicles are represented as ellipses
    with minimized volumes over the rectangular opponent shape"""

    def __init__(self, s_grid: np.ndarray,
                 param_handler: "ParamHandler",
                 params_ego: KinematicModelParameters,
                 params_opp: List[KinematicModelParameters]):
        """
        Initialize model
        :param s_grid: Longitudinal road coordinate grid in (m)
        :param p_kappa:  Parameter for curvature. Needs to be casadi symbolic.
        :param p_x_obst: Parameters for obstacle states as a list of casadi expressions
        :param params_ego: Ego model parameters
        :param params_opp: Parameters of opposing vehicles
        """
        super().__init__(s_grid, param_handler=param_handler, params=params_ego)

        assert param_handler.get_element("p_obst").width == len(params_opp)
        self.n_obstacles = len(params_opp)
        self.p_x_opp = param_handler.get_casadi("p_obst")
        self.p_norm = param_handler.get_casadi("p_norm")
        p_norm = self.p_norm

        self.safe_dist = []
        for i in range(self.n_obstacles):
            # parse parameters of opponent
            s0_opp = self.p_x_opp[0, i]
            n0_opp = self.p_x_opp[1, i]
            alpha_opp = self.p_x_opp[2, i]

            # compute main axes
            a = (params_opp[i].chassis_length / 2. + params_ego.safety_radius_length) * cs.power(2, 1 / p_norm)
            b = (params_opp[i].chassis_width / 2. + params_ego.safety_radius_side) * cs.power(2, 1 / p_norm)

            # rotation matrix
            R = cs.vertcat(cs.horzcat(cs.cos(alpha_opp), -cs.sin(alpha_opp)),
                           cs.horzcat(cs.sin(alpha_opp), cs.cos(alpha_opp)))

            # center matrix
            K = cs.vertcat(s0_opp + params_opp[i].length / 2 * cs.cos(alpha_opp),
                           n0_opp + params_opp[i].length / 2 * cs.sin(alpha_opp))

            # compact state
            center_ego = cs.vertcat(self.state_s + params_ego.length / 2 * cs.cos(self.state_alpha),
                                    self.state_n + params_ego.length / 2 * cs.sin(self.state_alpha))

            norm_vec = cs.transpose(R) @ (center_ego - K)
            norm_val = (cs.fabs(norm_vec[0]) / a) ** p_norm + (cs.fabs(norm_vec[1]) / b) ** p_norm
            norm_val = cs.power(norm_val, 1 / p_norm)

            condition = norm_val - 1

            self.safe_dist.append(condition)


class FrenetModelWithObstaclesPnormPanos(FrenetModel):
    """ Kinematic vehicle model with obstacles represented as p-norm
    The distance measure for the ego vehicle is a cicular distance, the opposing vehicles are represented as ellipses
    with minimized volumes over the rectangular opponent shape"""

    def __init__(self, s_grid: np.ndarray,
                 param_handler: "ParamHandler",
                 params_ego: KinematicModelParameters,
                 params_opp: List[KinematicModelParameters]):
        """
        Initialize model
        :param s_grid: Longitudinal road coordinate grid in (m)
        :param p_kappa:  Parameter for curvature. Needs to be casadi symbolic.
        :param p_x_obst: Parameters for obstacle states as a list of casadi expressions
        :param params_ego: Ego model parameters
        :param params_opp: Parameters of opposing vehicles
        """
        super().__init__(s_grid, param_handler=param_handler, params=params_ego)

        assert param_handler.get_element("p_obst").width == len(params_opp)
        self.n_obstacles = len(params_opp)
        self.p_x_opp = param_handler.get_casadi("p_obst")
        self.p_norm = param_handler.get_casadi("p_norm")
        p_norm = self.p_norm

        self.safe_dist = []
        for i in range(self.n_obstacles):
            # parse parameters of opponent
            s0_opp = self.p_x_opp[0, i]
            n0_opp = self.p_x_opp[1, i]
            alpha_opp = self.p_x_opp[2, i]

            # compute main axes
            a = (params_opp[i].chassis_length / 2. + params_ego.safety_radius_length)
            b = (params_opp[i].chassis_width / 2. + params_ego.safety_radius_side)

            # rotation matrix
            R = cs.vertcat(cs.horzcat(cs.cos(alpha_opp), -cs.sin(alpha_opp)),
                           cs.horzcat(cs.sin(alpha_opp), cs.cos(alpha_opp)))

            # center matrix
            K = cs.vertcat(s0_opp + params_opp[i].length / 2 * cs.cos(alpha_opp),
                           n0_opp + params_opp[i].length / 2 * cs.sin(alpha_opp))

            # compact state
            center_ego = cs.vertcat(self.state_s + params_ego.length / 2 * cs.cos(self.state_alpha),
                                    self.state_n + params_ego.length / 2 * cs.sin(self.state_alpha))

            norm_vec = cs.transpose(R) @ (center_ego - K)
            x = norm_vec[0] / a
            y = norm_vec[1] / b

            condition = cs.fmax(1 - x, 0) ** 2 * cs.fmax(x + 1, 0) ** 2 * cs.fmax(1 - y, 0) ** 2 * cs.fmax(y + 1,
                                                                                                           0) ** 2

            self.safe_dist.append(condition)
            self.safe_dist.append(-condition)


class FrenetModelWithObstaclesLogSumExp(FrenetModel):
    """ Kinematic vehicle model with obstacles represented as p-norm
    The distance measure for the ego vehicle is a cicular distance, the opposing vehicles are represented as ellipses
    with minimized volumes over the rectangular opponent shape"""

    def __init__(self, s_grid: np.ndarray,
                 param_handler: "ParamHandler",
                 params_ego: KinematicModelParameters,
                 params_opp: List[KinematicModelParameters]):
        """
        Initialize model
        :param s_grid: Longitudinal road coordinate grid in (m)
        :param p_kappa:  Parameter for curvature. Needs to be casadi symbolic.
        :param p_x_obst: Parameters for obstacle states as a list of casadi expressions
        :param params_ego: Ego model parameters
        :param params_opp: Parameters of opposing vehicles
        """
        super().__init__(s_grid, param_handler=param_handler, params=params_ego)

        assert param_handler.get_element("p_obst").width == len(params_opp)
        self.n_obstacles = len(params_opp)
        self.p_x_opp = param_handler.get_casadi("p_obst")
        self.p_norm = param_handler.get_casadi("p_norm")
        alpha = self.p_norm  # naming conflict but good for now

        x_eval, y_eval = 1, 1
        fac = 1 / alpha * np.log(0.25 * (
                np.exp(alpha * x_eval) + np.exp(-alpha * x_eval) + np.exp(alpha * y_eval) + np.exp(
            -alpha * y_eval)))

        self.safe_dist = []
        for i in range(self.n_obstacles):
            # parse parameters of opponent
            s0_opp = self.p_x_opp[0, i]
            n0_opp = self.p_x_opp[1, i]
            alpha_opp = self.p_x_opp[2, i]

            # compute main axes
            a = (params_opp[i].chassis_length / 2. + params_ego.safety_radius_length)
            b = (params_opp[i].chassis_width / 2. + params_ego.safety_radius_side)

            # rotation matrix
            R = cs.vertcat(cs.horzcat(cs.cos(alpha_opp), -cs.sin(alpha_opp)),
                           cs.horzcat(cs.sin(alpha_opp), cs.cos(alpha_opp)))

            # center matrix
            K = cs.vertcat(s0_opp + params_opp[i].length / 2 * cs.cos(alpha_opp),
                           n0_opp + params_opp[i].length / 2 * cs.sin(alpha_opp))

            # compact state
            center_ego = cs.vertcat(self.state_s + params_ego.length / 2 * cs.cos(self.state_alpha),
                                    self.state_n + params_ego.length / 2 * cs.sin(self.state_alpha))

            norm_vec = cs.transpose(R) @ (center_ego - K)
            x = norm_vec[0] / a
            y = norm_vec[1] / b
            impl_func = 1 / (alpha * fac) * (
                cs.log(0.25 * (cs.exp(alpha * x) + cs.exp(-alpha * x) + cs.exp(alpha * y) + cs.exp(-alpha * y))))

            condition = impl_func - 1

            self.safe_dist.append(condition)


class FrenetModelWithObstaclesBoltzmann(FrenetModel):
    """ Kinematic vehicle model with obstacles represented as p-norm
    The distance measure for the ego vehicle is a cicular distance, the opposing vehicles are represented as ellipses
    with minimized volumes over the rectangular opponent shape"""

    def __init__(self, s_grid: np.ndarray,
                 param_handler: "ParamHandler",
                 params_ego: KinematicModelParameters,
                 params_opp: List[KinematicModelParameters]):
        """
        Initialize model
        :param s_grid: Longitudinal road coordinate grid in (m)
        :param p_kappa:  Parameter for curvature. Needs to be casadi symbolic.
        :param p_x_obst: Parameters for obstacle states as a list of casadi expressions
        :param params_ego: Ego model parameters
        :param params_opp: Parameters of opposing vehicles
        """
        super().__init__(s_grid, param_handler=param_handler, params=params_ego)

        assert param_handler.get_element("p_obst").width == len(params_opp)
        self.n_obstacles = len(params_opp)
        self.p_x_opp = param_handler.get_casadi("p_obst")
        self.p_norm = param_handler.get_casadi("p_norm")
        alpha = self.p_norm  # naming conflict but good for now

        fac = (2 * np.exp(alpha) - 2 * np.exp(-alpha)) / (2 * np.exp(alpha) + 2 * np.exp(-alpha))

        self.safe_dist = []
        for i in range(self.n_obstacles):
            # parse parameters of opponent
            s0_opp = self.p_x_opp[0, i]
            n0_opp = self.p_x_opp[1, i]
            alpha_opp = self.p_x_opp[2, i]

            # compute main axes
            a = (params_opp[i].chassis_length / 2. + params_ego.safety_radius_length)
            b = (params_opp[i].chassis_width / 2. + params_ego.safety_radius_side)

            # rotation matrix
            R = cs.vertcat(cs.horzcat(cs.cos(alpha_opp), -cs.sin(alpha_opp)),
                           cs.horzcat(cs.sin(alpha_opp), cs.cos(alpha_opp)))

            # center matrix
            K = cs.vertcat(s0_opp + params_opp[i].length / 2 * cs.cos(alpha_opp),
                           n0_opp + params_opp[i].length / 2 * cs.sin(alpha_opp))

            # compact state
            center_ego = cs.vertcat(self.state_s + params_ego.length / 2 * cs.cos(self.state_alpha),
                                    self.state_n + params_ego.length / 2 * cs.sin(self.state_alpha))

            norm_vec = cs.transpose(R) @ (center_ego - K)
            x = norm_vec[0] / a
            y = norm_vec[1] / b
            impl_func = 1 / fac * (x * cs.exp(alpha * x) - x * cs.exp(-alpha * x) + y * cs.exp(alpha * y) - y * cs.exp(
                alpha * y)) / (cs.exp(alpha * x) + cs.exp(-alpha * x) + cs.exp(alpha * y) + cs.exp(
                -alpha * y))  # what is different than below???
            x0 = 0
            impl_func = 1 / fac * (
                    (x - x0) * cs.exp(alpha * (x - x0)) - (x - x0) * cs.exp(-alpha * (x - x0)) + y * cs.exp(
                alpha * y) - y * cs.exp(-alpha * y)) / (
                                cs.exp(alpha * (x - x0)) + cs.exp(-alpha * (x - x0)) + cs.exp(alpha * y) + cs.exp(
                            -alpha * y))

            condition = impl_func - 1

            self.safe_dist.append(condition)


class FrenetModelWithObstaclesEllipse(FrenetModel):
    """ Kinematic vehicle model with obstacles represented as ellipses
    https://www.geometrictools.com/Documentation/InformationAboutEllipses.pdf
    The distance measure for the ego vehicle is a cicular distance, the opposing vehicles are represented as ellipses
    with minimized volumes over the rectangular opponent shape"""

    def __init__(self, s_grid: np.ndarray,
                 param_handler: "ParamHandler",
                 params_ego: KinematicModelParameters,
                 params_opp: List[KinematicModelParameters]):
        """
        Initialize model
        :param s_grid: Longitudinal road coordinate grid in (m)
        :param p_kappa:  Parameter for curvature. Needs to be casadi symbolic.
        :param p_x_obst: Parameters for obstacle states as a list of casadi expressions
        :param params_ego: Ego model parameters
        :param params_opp: Parameters of opposing vehicles
        """
        super().__init__(s_grid, param_handler=param_handler, params=params_ego)

        assert param_handler.get_element("p_obst").width == len(params_opp)
        self.n_obstacles = len(params_opp)
        self.p_x_opp = param_handler.get_casadi("p_obst")

        self.safe_dist = []
        for i in range(self.n_obstacles):
            # parse parameters of opponent
            s0_opp = self.p_x_opp[0, i]
            n0_opp = self.p_x_opp[1, i]
            alpha_opp = self.p_x_opp[2, i]

            # compute main axes of ellipse
            a = params_opp[i].chassis_length / np.sqrt(2) + params_ego.safety_radius_length
            b = params_opp[i].chassis_width / np.sqrt(2) + params_ego.safety_radius_side

            # main axis matrix
            D = np.array([[1 / (a ** 2), 0],
                          [0, 1 / (b ** 2)]])

            # rotation matrix
            R = cs.vertcat(cs.horzcat(cs.cos(alpha_opp), -cs.sin(alpha_opp)),
                           cs.horzcat(cs.sin(alpha_opp), cs.cos(alpha_opp)))

            # center matrix
            K = cs.vertcat(s0_opp + params_opp[i].length / 2 * cs.cos(alpha_opp),
                           n0_opp + params_opp[i].length / 2 * cs.sin(alpha_opp))

            # compact state
            center_ego = cs.vertcat(self.state_s + params_ego.length / 2 * cs.cos(self.state_alpha),
                                    self.state_n + params_ego.length / 2 * cs.sin(self.state_alpha))

            condition = cs.sqrt(cs.transpose(center_ego - K) @ R @ D @ cs.transpose(R) @ (center_ego - K)) - 1

            self.safe_dist.append(condition)


class FrenetModelCG(KinematicVehicleModelFrenet):
    """ Basic most standard kinematic vehicle model
    Source: Kloeser et al."""

    def __init__(self, s_grid: np.ndarray, p_kappa, p_nl, p_nr, params: KinematicModelParameters):
        super().__init__(s_grid, p_kappa, p_nl, p_nr, params)

        s_dot = self.state_v * cs.cos(self.state_alpha + self.beta) / (
                1 - self.state_n * self.interp_s2kappa(self.state_s, p_kappa))
        self.rhs[0] = s_dot
        self.rhs[1] = self.state_v * cs.sin(self.state_alpha + self.beta)
        self.rhs[2] = self.state_v / self.lr * cs.sin(self.beta) - s_dot * self.interp_s2kappa(self.state_s,
                                                                                               p_kappa)

        self.post_init()


class FrenetModelCG2(KinematicVehicleModelFrenet):
    """ Basic most standard kinematic vehicle model
    Source: https://www.researchgate.net/publication/301275605_A_Hierarchical_Model_Predictive_Control_Framework_for_On-
    road_Formation_Control_of_Autonomous_Vehicles/download"""

    def __init__(self, s_grid: np.ndarray, p_kappa, p_nl, p_nr, params: KinematicModelParameters):
        super().__init__(s_grid, p_kappa, p_nl, p_nr, params)
        # dx/dt = f(x,u)
        lr = self.lr
        lf = self.lf
        beta = cs.arctan(lr / (lr + lf) * cs.tan(self.delta))
        s_dot = self.state_v * cs.cos(self.state_alpha) / (
                1 - self.state_n * self.interp_s2kappa(self.state_s, self.p_kappa))
        self.rhs[0] = s_dot
        self.rhs[1] = self.state_v * cs.sin(self.state_alpha)
        self.rhs[2] = self.state_v / lr * cs.sin(beta) - \
                      s_dot * self.interp_s2kappa(self.state_s, self.p_kappa) \
                      + lr / (lr + lf) * self.delta
        self.post_init()


class SimpleFrenetModel(KinematicVehicleModelFrenet):
    """ Simplified version of standard kinematic vehicle model. Mostly simplified trigonometric functions """

    def __init__(self, s_grid: np.ndarray, p_kappa, p_nl, p_nr, params: KinematicModelParameters):
        super().__init__(s_grid, p_kappa, p_nl, p_nr, params)
        length = params.length_front + params.length_rear
        s_dot = self.state_v * cs.cos(self.state_alpha) / (
                1 - self.state_n * self.interp_s2kappa(self.state_s, self.p_kappa))
        self.rhs[0] = s_dot
        self.rhs[1] = self.state_v * self.state_alpha
        self.rhs[2] = self.state_v * self.state_delta / length - \
                      s_dot * self.interp_s2kappa(self.state_s, self.p_kappa)

        self.post_init()


class PointMassFrenetModel:
    """simple most frenet coordinate model resembling a point mass traveling in Frenet coordinates"""

    def __init__(self, s_grid: np.ndarray, p_kappa, p_nl, p_nr, params: KinematicModelParameters):
        self.params_ = params
        self.rhs = None
        self.f_ode = None
        self.name = None

        self.x_states = cs.MX.sym('x_states', 4)
        self.u_controls = cs.MX.sym('u_controls', 2)

        self.state_s = self.x_states[0]
        self.state_n = self.x_states[1]
        self.state_vs = self.x_states[2]
        self.state_vn = self.x_states[3]

        self.control_Fx = self.u_controls[0]
        self.control_Fy = self.u_controls[1]

        self.interp_s2nl = cs.interpolant("interp_s2nl", "linear", [s_grid])
        self.interp_s2nr = cs.interpolant("interp_s2nr", "linear", [s_grid])

        if type(p_kappa) is np.ndarray:
            # the parameter kappa is non symbolic
            interp_s2kappa_tmp = cs.interpolant("interp_s2kappa", "linear", [s_grid], p_kappa)
            self.interp_s2kappa = lambda s, p: interp_s2kappa_tmp(s)  # we need the standard interface of 2 parameters
        else:  # must be casadi type
            # the parameter kappa is symbolic and can be used as paramter in optimization problem
            self.interp_s2kappa = cs.interpolant("interp_s2kappa", "linear", [s_grid])

        s_dot = self.state_vs / (1 - self.state_n * self.interp_s2kappa(self.state_s, p_kappa))
        self.rhs = [s_dot,
                    self.state_vn,
                    1 / params.mass * self.control_Fx,
                    1 / params.mass * self.control_Fy]

        # Lateral acceleration at rear wheel
        radius_eff = 1 / (self.interp_s2kappa(self.state_s, p_kappa)) - self.state_n
        self.a_lat = self.state_vs * self.state_vs * 1 / radius_eff \
                     + 1 / params.mass * self.control_Fy
        self.f_a_lat = cs.Function('f_a_lat', [self.x_states, self.u_controls], [self.a_lat])

        self.rhs = cs.vertcat(*self.rhs)
        self.f_ode = cs.Function('f_ode', [self.x_states, self.u_controls], [self.rhs])
        self.n_x = 4
