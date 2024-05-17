from typing import List
from vehicle_models.utils import get_circle_rect_cover, halfspace
import casadi as cs
import numpy as np
from acados_template import AcadosModel
from vehicle_models.model_kinematic import (KinematicVehicleModel,
                                            KinematicModelParameters)
from vehicle_models.model_kinematic_frenet import KinematicVehicleModelFrenet


class DualModel(KinematicVehicleModelFrenet):
    """ Basic most standard kinematic vehicle model """

    def __init__(self, s_grid: np.ndarray, p_kappa, params: KinematicModelParameters):
        super().__init__(s_grid, p_kappa, params, add_states=3)

        # Standard frenet model part
        length = params.length_front + params.length_rear
        s_dot = self.state_v * cs.cos(self.state_alpha) / (
                1 - self.state_n * self.interp_s2kappa(self.state_s, p_kappa))
        self.rhs[0] = s_dot
        self.rhs[1] = self.state_v * cs.sin(self.state_alpha)
        self.rhs[2] = self.state_v / length * cs.tan(self.state_delta) - s_dot * self.interp_s2kappa(
            self.state_s, p_kappa)

        # additional cartesian part
        self.state_x = self.x_states[-3]
        self.state_y = self.x_states[-2]
        self.state_theta = self.x_states[-1]

        self.rhs[-3] = self.state_v * cs.cos(self.state_theta)
        self.rhs[-2] = self.state_v * cs.sin(self.state_theta)
        self.rhs[-1] = self.state_v * cs.tan(self.state_delta) / params.length

        self.post_init()


class DualModelWithObstaclesBasic(DualModel):
    """ Basic most standard kinematic vehicle model with obstacles
    The obstacle avoidance is modelled by means of the distance of two circles."""

    def __init__(self, s_grid: np.ndarray,
                 p_kappa: "curvature parameter as casadi parameters",
                 p_x_obst: List,
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
        super().__init__(s_grid, p_kappa, params_ego)

        if hasattr(self, "state_x") and hasattr(self, "state_y") and hasattr(self, "state_theta"):
            state_x = self.state_x
            state_y = self.state_y
            theta = self.state_theta
        elif hasattr(self, "state_s") and hasattr(self, "state_n") and hasattr(self, "state_alpha"):
            state_x = self.state_s
            state_y = self.state_n
            theta = self.state_alpha
        else:
            state_x = None
            state_y = None
            theta = None
            Exception()

        assert len(p_x_obst) == len(params_opp)
        self.n_obstacles = len(params_opp)
        self.p_x_opp = p_x_obst

        self.safe_dist = []
        for i in range(self.n_obstacles):
            center_dist_sqr = (state_x - self.p_x_opp[i][0]) ** 2 + (state_y - self.p_x_opp[i][1]) ** 2
            self.safe_dist.append(center_dist_sqr - (params_opp[i].safety_radius_side + params_ego.safety_radius_side) ** 2)


class DualModelWithObstaclesEllipse(DualModel):
    """ Kinematic vehicle model with obstacles represented as ellipses
    https://www.geometrictools.com/Documentation/InformationAboutEllipses.pdf
    The distance measure for the ego vehicle is a cicular distance, the opposing vehicles are represented as ellipses
    with minimized volumes over the rectangular opponent shape"""

    def __init__(self, s_grid: np.ndarray,
                 p_kappa: "curvature parameter as casadi parameters",
                 p_x_obst: List,
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
        super().__init__(s_grid, p_kappa, params_ego)

        if hasattr(self, "state_x") and hasattr(self, "state_y") and hasattr(self, "state_theta"):
            state_x = self.state_x
            state_y = self.state_y
            theta = self.state_theta
        elif hasattr(self, "state_s") and hasattr(self, "state_n") and hasattr(self, "state_alpha"):
            state_x = self.state_s
            state_y = self.state_n
            theta = self.state_alpha
        else:
            state_x = None
            state_y = None
            theta = None
            Exception()

        assert len(p_x_obst) == len(params_opp)
        self.n_obstacles = len(params_opp)
        self.p_x_opp = p_x_obst

        self.safe_dist = []
        for i in range(self.n_obstacles):
            # parse parameters of opponent
            s0_opp = p_x_obst[i][0]
            n0_opp = p_x_obst[i][1]
            alpha_opp = p_x_obst[i][2]

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
            center_ego = cs.vertcat(state_x + params_ego.length / 2 * cs.cos(theta),
                                    state_y + params_ego.length / 2 * cs.sin(theta))

            condition = cs.transpose(center_ego - K) @ R @ D @ cs.transpose(R) @ (center_ego - K) - 1

            self.safe_dist.append(condition)


class DualModelWithObstaclesCircles(DualModel):
    """ Kinematic vehicle model with circular obstacles. Each obstacle and ego vehicle is modelled by a number of
    circles. Formulated with square root expression due to numerical improvement. """

    def __init__(self, s_grid: np.ndarray,
                 p_kappa: "curvature parameter as casadi parameters",
                 p_x_obst: List,
                 params_ego: KinematicModelParameters,
                 params_opp: List[KinematicModelParameters],
                 n_circles_ego: int = 1,
                 n_circles_opp: int = 1,
                 auto_size: bool = False):
        """
        Initialize model
        :param s_grid: Longitudinal road coordinate grid in (m)
        :param p_kappa:  Parameter for curvature. Needs to be casadi symbolic.
        :param p_x_obst: Parameters for obstacle states as a list of casadi expressions
        :param params_ego: Ego model parameters
        :param params_opp: Parameters of opposing vehicles
        :param n_circles_ego: number of circles the ego model is modelled with
        :param n_circles_opp: number of circles the opponent model is modelled with
        :param auto_size: automatically fit smallest circles over vehicle, otherwise use safty dist. parameter
        """

        super().__init__(s_grid, p_kappa, params_ego)
        assert len(p_x_obst) == len(params_opp)

        if hasattr(self, "state_x") and hasattr(self, "state_y") and hasattr(self, "state_theta"):
            state_x = self.state_x
            state_y = self.state_y
            theta = self.state_theta
        elif hasattr(self, "state_s") and hasattr(self, "state_n") and hasattr(self, "state_alpha"):
            state_x = self.state_s
            state_y = self.state_n
            theta = self.state_alpha
        else:
            state_x = None
            state_y = None
            theta = None
            Exception()

        self.n_obstacles = len(params_opp)
        self.p_x_opp = p_x_obst
        centers_s_opp = []
        centers_n_opp = []
        radii_opp = []

        centers_s_ego, centers_n_ego, radius_ego = get_circle_rect_cover(n_circ=n_circles_ego,
                                                                         wheelbase=params_ego.length,
                                                                         l_rect=params_ego.chassis_length,
                                                                         w_rect=params_ego.chassis_width,
                                                                         state_s=state_x,
                                                                         state_n=state_y,
                                                                         state_alpha=theta)

        for i in range(self.n_obstacles):
            center_s_opp, center_n_opp, radius_opp = get_circle_rect_cover(n_circ=n_circles_opp,
                                                                           wheelbase=params_ego.length,
                                                                           l_rect=params_opp[i].chassis_length,
                                                                           w_rect=params_opp[i].chassis_width,
                                                                           state_s=p_x_obst[i][0],
                                                                           state_n=p_x_obst[i][1],
                                                                           state_alpha=p_x_obst[i][2])
            centers_s_opp.append(center_s_opp)
            centers_n_opp.append(center_n_opp)
            radii_opp.append(radius_opp)

        self.safe_dist = []

        for i_obstacle in range(self.n_obstacles):
            for i_ego_circ in range(n_circles_ego):
                for i_opp_circ in range(n_circles_opp):
                    s_dist_sqr = (centers_s_ego[i_ego_circ] - centers_s_opp[i_obstacle][i_opp_circ]) ** 2
                    n_dist_sqr = (centers_n_ego[i_ego_circ] - centers_n_opp[i_obstacle][i_opp_circ]) ** 2

                    center_dist_sqr = s_dist_sqr + n_dist_sqr
                    center_dist = cs.sqrt(center_dist_sqr)
                    # Either use exact distance to fit rectangle or defined in parameters
                    if auto_size:
                        min_dist_sqr = (radius_ego + radii_opp[i_obstacle])
                    else:
                        min_dist_sqr = (params_ego.safety_radius_side + params_opp[i_obstacle].safety_radius_side)
                    self.safe_dist.append(center_dist - min_dist_sqr)


class DualModelWithObstaclesPanos(DualModel):
    """
    Kinematic vehicle model with obstacles represented as hyperplanes
    "Embedded nonlinear model predictive control for obstacle avoidance
    using PANOC"-Sathya, et al., ECC2018
    """

    def __init__(self, s_grid: np.ndarray,
                 p_kappa: "curvature parameter as casadi parameters",
                 p_x_obst: List,
                 params_ego: KinematicModelParameters,
                 params_opp: List[KinematicModelParameters],
                 logmaxexp_par: float = 0.1,
                 increase_opp_size: float = 0.):
        """
        Initialize model
        :param s_grid: Longitudinal road coordinate grid in (m)
        :param p_kappa:  Parameter for curvature. Needs to be casadi symbolic.
        :param p_x_obst: Parameters for obstacle states as a list of casadi expressions
        :param params_ego: Ego model parameters
        :param params_opp: Parameters of opposing vehicles
        """
        super().__init__(s_grid, p_kappa, params_ego)

        assert len(p_x_obst) == len(params_opp)

        if hasattr(self, "state_x") and hasattr(self, "state_y") and hasattr(self, "state_theta"):
            state_x = self.state_x
            state_y = self.state_y
            theta = self.state_theta
        elif hasattr(self, "state_s") and hasattr(self, "state_n") and hasattr(self, "state_alpha"):
            state_x = self.state_s
            state_y = self.state_n
            theta = self.state_alpha
        else:
            state_x = None
            state_y = None
            theta = None
            Exception()

        self.n_obstacles = len(params_opp)
        self.p_x_opp = p_x_obst

        # compute test points of ego vehicle
        s_middle = state_x + params_ego.length / 2 * cs.cos(theta)
        n_middle = state_y + params_ego.length / 2 * cs.sin(theta)

        # 2x4 matrix
        offset = cs.vertcat(cs.horzcat(s_middle, s_middle, s_middle, s_middle),
                            cs.horzcat(n_middle, n_middle, n_middle, n_middle))

        # 2x4 matrix
        vertices = np.transpose(np.array([[params_ego.chassis_length / 2, params_ego.chassis_width / 2],
                                          [-params_ego.chassis_length / 2, params_ego.chassis_width / 2],
                                          [-params_ego.chassis_length / 2, -params_ego.chassis_width / 2],
                                          [params_ego.chassis_length / 2, -params_ego.chassis_width / 2]]))

        # 2x2 rotation matrix
        rotation_mat = cs.vertcat(cs.horzcat(cs.cos(theta), -cs.sin(theta)),
                                  cs.horzcat(cs.sin(theta), cs.cos(theta)))

        test_vertices = rotation_mat @ vertices + offset

        self.safe_dist = []
        vertex_order = [0, 1, 2, 3, 0]
        for i in range(self.n_obstacles):
            # parse parameters of opponent
            s0_opp = p_x_obst[i][0]
            n0_opp = p_x_obst[i][1]
            alpha_opp = p_x_obst[i][2]

            # compute test points of opp vehicle
            s_middle = s0_opp + params_opp[i].length / 2 * cs.cos(alpha_opp)
            n_middle = n0_opp + params_opp[i].length / 2 * cs.sin(alpha_opp)

            # rotation matrix 2x2
            rotation_mat = cs.vertcat(cs.horzcat(cs.cos(alpha_opp), -cs.sin(alpha_opp)),
                                      cs.horzcat(cs.sin(alpha_opp), cs.cos(alpha_opp)))

            # 2x4 matrix
            offset = cs.vertcat(cs.horzcat(s_middle, s_middle, s_middle, s_middle),
                                cs.horzcat(n_middle, n_middle, n_middle, n_middle))

            # 2x4 matrix
            half_length = params_opp[i].chassis_length / 2 + increase_opp_size
            half_width = params_opp[i].chassis_width / 2 + increase_opp_size
            vertices = np.transpose(np.array([[half_length, half_width],
                                              [-half_length, half_width],
                                              [-half_length, -half_width],
                                              [half_length, -half_width]]))

            # 2x4 matrix
            opponent_vertices = rotation_mat @ vertices + offset

            for i_vertice in range(test_vertices.shape[1]):
                x_vertex_test = test_vertices[0, i_vertice]
                y_vertex_test = test_vertices[1, i_vertice]
                half_space_1 = 1.
                for i_opp_vertex, i_opp_vertex_next in zip(vertex_order[:-1], vertex_order[1:]):
                    a, b, c = halfspace(x1=opponent_vertices[0, i_opp_vertex],
                                        y1=opponent_vertices[1, i_opp_vertex],
                                        x2=opponent_vertices[0, i_opp_vertex_next],
                                        y2=opponent_vertices[1, i_opp_vertex_next])
                    hyperplane = x_vertex_test * a + y_vertex_test * b + c
                    half_space_1 *= cs.log(cs.exp(logmaxexp_par * hyperplane) + 1) ** 2

                self.safe_dist.append(half_space_1)
                self.safe_dist.append(-half_space_1)


class DualModelWithObstaclesHyperplane(DualModel):
    """
    Finding the separating hyperplane
    """

    def __init__(self, s_grid: np.ndarray,
                 p_kappa: "curvature parameter as casadi parameters",
                 p_x_obst: List,
                 params_ego: KinematicModelParameters,
                 params_opp: List[KinematicModelParameters],
                 increase_opp_size: float = 0.):
        """
        Initialize model
        :param s_grid: Longitudinal road coordinate grid in (m)
        :param p_kappa:  Parameter for curvature. Needs to be casadi symbolic.
        :param p_x_obst: Parameters for obstacle states as a list of casadi expressions
        :param params_ego: Ego model parameters
        :param params_opp: Parameters of opposing vehicles
        :param increase_opp_size: increase the opponent by this value in each direction
        """
        super().__init__(s_grid, p_kappa, params_ego)

        assert len(p_x_obst) == len(params_opp)

        if hasattr(self, "state_x") and hasattr(self, "state_y") and hasattr(self, "state_theta"):
            state_x = self.state_x
            state_y = self.state_y
            theta = self.state_theta
        elif hasattr(self, "state_s") and hasattr(self, "state_n") and hasattr(self, "state_alpha"):
            state_x = self.state_s
            state_y = self.state_n
            theta = self.state_alpha
        else:
            state_x = None
            state_y = None
            theta = None
            Exception()

        self.n_obstacles = len(params_opp)
        self.p_x_opp = p_x_obst

        # compute test points of ego vehicle
        s_middle = state_x + params_ego.length / 2 * cs.cos(theta)
        n_middle = state_y + params_ego.length / 2 * cs.sin(theta)

        # 2x4 matrix
        offset = cs.vertcat(cs.horzcat(s_middle, s_middle, s_middle, s_middle),
                            cs.horzcat(n_middle, n_middle, n_middle, n_middle))

        # 2x4 matrix
        vertices = np.transpose(np.array([[params_ego.chassis_length / 2, params_ego.chassis_width / 2],
                                          [-params_ego.chassis_length / 2, params_ego.chassis_width / 2],
                                          [-params_ego.chassis_length / 2, -params_ego.chassis_width / 2],
                                          [params_ego.chassis_length / 2, -params_ego.chassis_width / 2]]))

        # 2x2 rotation matrix
        rotation_mat = cs.vertcat(cs.horzcat(cs.cos(theta), -cs.sin(theta)),
                                  cs.horzcat(cs.sin(theta), cs.cos(theta)))

        ego_vertices = rotation_mat @ vertices + offset

        self.safe_dist = []
        for i in range(self.n_obstacles):
            # parse parameters of opponent
            s0_opp = p_x_obst[i][0]
            n0_opp = p_x_obst[i][1]
            alpha_opp = p_x_obst[i][2]

            # compute test points of opp vehicle
            s_middle = s0_opp + params_opp[i].length / 2 * cs.cos(alpha_opp)
            n_middle = n0_opp + params_opp[i].length / 2 * cs.sin(alpha_opp)

            # rotation matrix 2x2
            rotation_mat = cs.vertcat(cs.horzcat(cs.cos(alpha_opp), -cs.sin(alpha_opp)),
                                      cs.horzcat(cs.sin(alpha_opp), cs.cos(alpha_opp)))

            # 2x4 matrix
            offset = cs.vertcat(cs.horzcat(s_middle, s_middle, s_middle, s_middle),
                                cs.horzcat(n_middle, n_middle, n_middle, n_middle))

            # 2x4 matrix
            half_length = params_opp[i].chassis_length / 2 + increase_opp_size
            half_width = params_opp[i].chassis_width / 2 + increase_opp_size
            vertices = np.transpose(np.array([[half_length, half_width],
                                              [-half_length, half_width],
                                              [-half_length, -half_width],
                                              [half_length, -half_width]]))

            # 2x4 matrix
            opp_vertices = rotation_mat @ vertices + offset

            # hyperplane parameters
            a = cs.MX.sym('a_' + str(i), 1)
            b = cs.MX.sym('b_' + str(i), 1)
            c = cs.MX.sym('c_' + str(i), 1)

            self.u_controls = cs.vertcat(self.u_controls, a)
            self.u_controls = cs.vertcat(self.u_controls, b)
            self.u_controls = cs.vertcat(self.u_controls, c)

            # constrain search space in order to improve solution
            self.safe_dist.append(c)
            self.safe_dist.append(a + 1.)
            self.safe_dist.append(-a + 1.)
            self.safe_dist.append(b + 1.)
            self.safe_dist.append(-b + 1.)


            # constraint for solutions not degenerating
            self.safe_dist.append((a ** 2 + b ** 2) - 1)
            self.safe_dist.append(-(a ** 2 + b ** 2) + 1)

            for i_vertex in range(4):
                # hyperplane constraints. Either on one or the other side
                ego_condition = a * ego_vertices[0, i_vertex] + b * ego_vertices[1, i_vertex] + c
                opp_condition = -(a * opp_vertices[0, i_vertex] + b * opp_vertices[1, i_vertex] + c)

                self.safe_dist.append(ego_condition)
                self.safe_dist.append(opp_condition)