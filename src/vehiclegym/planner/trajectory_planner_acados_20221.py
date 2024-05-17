import tempfile
from tempfile import mkdtemp
from logging import getLogger
import time
from copy import copy, deepcopy
import casadi as cs
import numpy as np
import scipy
from typing import List
from vehiclegym.planner.predicitor import Predictor
from acados_template import AcadosOcp, AcadosOcpSolver

from vehicle_models.model_kinematic import KinematicModelParameters
from vehicle_models.model_kinematic_frenet import FrenetModelWithObstaclesEllipse
from vehiclegym.utils.automotive_datastructures import FrenetTrajectory
from vehiclegym.road.road import RoadOptions, Road
from vehiclegym.planner.trajectory_planner_base import (
    PlannerOptions,
    Planner,
    get_time_grid, get_time_steps,
)
from vehiclegym.utils.acados_param_handler import ParamHandler

number_acados_planners = 0

logger = getLogger(__name__)


class VehiclePlannerAcados20221(Planner):
    """
    Acados planner class. This class is used to create the planner object based on the NLP solver acados
    """

    def __init__(
            self,
            ego_model_params: KinematicModelParameters,
            road: Road,
            planner_options: PlannerOptions,
            opp_model_params: List[KinematicModelParameters] = None,
    ):
        """
        Initialize the planner with the ego vehicle model parameters, road parameters and options
        :param ego_model_params: ego vehicle model parameters
        :param road: road definition class
        :param planner_options: planner options
        """

        # Super class init
        super().__init__()

        # create acados ocp
        ocp = AcadosOcp()

        # create render arguments
        self.planner_opts = planner_options
        self.time_disc_base = planner_options.time_disc
        self.time_disc = planner_options.time_disc  # todo: remove this
        self.nodes = planner_options.n_nodes
        self.n_nodes = planner_options.n_nodes
        self.states_ego = None
        self.is_road_circular = hasattr(road, "track_len")
        self.computation_time = -1
        global number_acados_planners
        number_acados_planners += 1

        self.param_handler = ParamHandler(acados_ocp=ocp, num_nodes=self.n_nodes)

        # define default configuration
        if opp_model_params is None:
            opp_model_params = [copy(ego_model_params), copy(ego_model_params)]
        self.opp_model_params = opp_model_params
        self.road = road
        self.deactivate_obstacle_avoidance = (
            planner_options.deactivate_obstacle_avoidance
        )

        # create parameter for curvature
        self.param_handler.add("p_kappa", road.s_grid_.shape[0])
        self.param_handler.add("p_nl", road.s_grid_.shape[0])
        self.param_handler.add("p_nr", road.s_grid_.shape[0])

        # obtain obstacle number and create parameters
        self.max_number_obstacles = len(opp_model_params)

        self.x_opposing_containers = []
        self.param_handler.add("p_obst", length=3, width=self.max_number_obstacles)
        for i in range(self.max_number_obstacles):
            self.x_opposing_containers.append(None)

        self.model = FrenetModelWithObstaclesEllipse(
            s_grid=road.s_grid_,
            param_handler=self.param_handler,
            params_ego=ego_model_params,
            params_opp=opp_model_params,
        )

        # set model
        acados_model, constraints_acc, constraints_boundary, constraints_obst = self.model.export4acados()
        acados_model.name = "planner_acados_" + str(number_acados_planners)
        self.acados_model = acados_model
        ocp.model = acados_model
        ocp.code_export_directory = tempfile.mkdtemp()

        # define constraint
        ocp.model.con_h_expr = cs.vertcat(constraints_acc, constraints_boundary, constraints_obst)
        ocp.model.con_h_expr_e = cs.vertcat(constraints_boundary, constraints_obst)

        # get constraint parameters
        self.n_con = ocp.model.con_h_expr.shape[0]
        self.n_con_acc = constraints_acc.shape[0]
        self.n_con_boundary = constraints_boundary.shape[0]
        self.n_con_obst = constraints_obst.shape[0]
        self.n_con_e = self.n_con_boundary + self.n_con_obst

        # set dimensions
        nx = acados_model.x.size()[0]
        nu = acados_model.u.size()[0]
        npar = acados_model.p.size()[0]
        self.npar, self.nu, self.nx = npar, nu, nx

        # weirdly acados is very sensitive to the obstacle state, that should not be considered
        self.out_of_track_state = np.array([50.0, 20, 0.0, 0.01, 0.0])

        ny = nx + nu
        ny_e = nx
        self.ny, self.ny_e = ny, ny_e

        ocp.dims.N = planner_options.n_nodes - 1

        ns = self.n_con  # We just constrain the constraints in the constraint-expression
        nsh = self.n_con
        self.current_slacks = np.zeros((ns,))

        # set cost
        Q = np.diag(planner_options.q0)
        R = np.diag(planner_options.r0)
        Qe = np.diag(planner_options.qn)
        self.Q, self.R = Q, R

        ocp.cost.cost_type = "LINEAR_LS"
        ocp.cost.cost_type_e = "LINEAR_LS"

        ocp.cost.W = scipy.linalg.block_diag(Q, R)
        ocp.cost.W_e = Qe

        ocp.cost.Vx = np.zeros((ny, nx))
        ocp.cost.Vx[:nx, :nx] = np.eye(nx)

        Vu = np.zeros((ny, nu))
        Vu[nx: nx + nu, :] = np.eye(nu)
        ocp.cost.Vu = Vu

        Vx_e = np.zeros((ny_e, nx))
        Vx_e[:nx, :nx] = np.eye(nx)
        ocp.cost.Vx_e = Vx_e

        ocp.cost.zl = np.array([planner_options.w_acc_slack_l1] * self.n_con_acc +
                               [planner_options.w_bound_slack_l1] * self.n_con_boundary +
                               [planner_options.w_obst_slack_l1] * self.n_con_obst)
        ocp.cost.Zl = np.array([planner_options.w_acc_slack_l2] * self.n_con_acc +
                               [planner_options.w_bound_slack_l2] * self.n_con_boundary +
                               [planner_options.w_obst_slack_l2] * self.n_con_obst)
        ocp.cost.zu, ocp.cost.Zu = ocp.cost.zl, ocp.cost.Zl

        ocp.cost.zl_e = np.array([planner_options.w_bound_slack_l1] * self.n_con_boundary +
                                 [planner_options.w_obst_slack_l1] * self.n_con_obst)
        ocp.cost.Zl_e = np.array([planner_options.w_bound_slack_l2] * self.n_con_boundary +
                                 [planner_options.w_obst_slack_l2] * self.n_con_obst)
        ocp.cost.zu_e, ocp.cost.Zu_e = ocp.cost.zl_e, ocp.cost.Zl_e

        # set intial references
        ocp.cost.yref = np.zeros((ny,))
        ocp.cost.yref_e = np.zeros((ny_e,))

        # parameter values
        ocp.parameter_values = np.zeros((npar,))

        # setting constraints
        ocp.constraints.lbx = np.array([-ego_model_params.maximum_alpha, 0])
        ocp.constraints.ubx = np.array(
            [ego_model_params.maximum_alpha, self.model.params_.maximum_velocity]
        )
        ocp.constraints.idxbx = np.array([2, 3])

        ocp.constraints.lbx_e = np.array([-ego_model_params.maximum_alpha, 0])
        ocp.constraints.ubx_e = np.array(
            [ego_model_params.maximum_alpha, self.planner_opts.v_max_terminal_set]
        )
        ocp.constraints.idxbx_e = np.array([2, 3])

        ocp.constraints.lbu = np.array(
            [
                -self.model.params_.maximum_deceleration_force,
                -self.model.params_.maximum_steering_angle,
            ]
        )
        ocp.constraints.ubu = np.array(
            [
                self.model.params_.maximum_acceleration_force,
                self.model.params_.maximum_steering_angle,
            ]
        )

        ocp.constraints.idxbu = np.array([0, 1])

        ocp.constraints.lh = np.zeros((self.n_con,))
        ocp.constraints.lh[0:self.n_con_acc] = np.array(
            [-self.model.params_.maximum_lateral_acc]
        )

        ocp.constraints.uh = np.ones((self.n_con,)) * 1e6
        ocp.constraints.uh[0:self.n_con_acc] = np.array(
            [self.model.params_.maximum_lateral_acc]
        )

        ocp.constraints.lh_e = ocp.constraints.lh[self.n_con_acc:]
        ocp.constraints.uh_e = ocp.constraints.uh[self.n_con_acc:]

        ocp.constraints.lsh = np.zeros(nsh)
        ocp.constraints.ush = np.zeros(nsh)
        ocp.constraints.idxsh = np.array(range(self.n_con))

        ocp.constraints.lsh_e = np.zeros(self.n_con_e)
        ocp.constraints.ush_e = np.zeros(self.n_con_e)
        ocp.constraints.idxsh_e = np.array(range(self.n_con_e))

        # solver values
        ocp.constraints.x0 = np.zeros((nx,))
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.qp_solver_cond_N = int(
            planner_options.n_nodes * planner_options.condensing_relative
        )
        ocp.solver_options.tol = (
            planner_options.qp_tolerance
        )  # 1e-2 is default, in RTI relevant for QP solution
        ocp.solver_options.nlp_solver_type = "SQP_RTI"  # SQP_RTI
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"  # "GAUSS_NEWTON"  # "EXACT"
        ocp.solver_options.integrator_type = "IRK"
        # ocp.solver_options.exact_hess_constr = 0
        ocp.solver_options.nlp_solver_max_iter = 1
        ocp.solver_options.sim_method_num_stages = 1
        ocp.solver_options.sim_method_num_steps = 2
        ocp.solver_options.time_steps = get_time_steps(
            n_nodes=planner_options.n_nodes,
            time_disc=planner_options.time_disc,
            time_stretch_fac=planner_options.time_stretch_fac)
        self.time_grid = get_time_grid(
            n_nodes=planner_options.n_nodes,
            time_disc=planner_options.time_disc,
            time_stretch_fac=planner_options.time_stretch_fac)

        ocp.solver_options.tf = self.time_grid[-1]
        self.time_pred = self.time_grid[-1]

        # print("Final time:{}".format(ocp.solver_options.tf))

        # initialize predictor
        self.predictor = Predictor(
            road=road,
            opponent_params=opp_model_params,
            ego_model=self.model,
            t_grid=self.time_grid,
        )

        # create solver
        if planner_options.use_cython:
            AcadosOcpSolver.generate(ocp, json_file="acados_ocp.json")
            AcadosOcpSolver.build(ocp.code_export_directory, with_cython=True)
            self.solver = AcadosOcpSolver.create_cython_solver("acados_ocp.json")
        else:
            self.solver = AcadosOcpSolver(
                ocp, json_file=ocp.code_export_directory + "/acados_ocp.json")
        self.param_handler.set_solver(self.solver)

        # set kappa values
        self.set_road(
            s_grid=road.s_grid_,
            kappa_grid=road.kappa_grid_,
            nl_grid=road.nl_grid_,
            nr_grid=road.nr_grid_,
        )
        initial_obstacle_pos = []
        for i in range(self.max_number_obstacles):
            initial_obstacle_pos.append(self.out_of_track_state)
        self.set_obstacles(obstacles=initial_obstacle_pos)

        # store initial guess
        self.ini_iterate_file = ocp.code_export_directory + "/ini_iter_solv" + str(number_acados_planners) + ".json"
        self.solver.store_iterate(filename=self.ini_iterate_file, overwrite=True)

        # enforce terminal boundary earlier
        times_rev = np.flip(
            np.hstack((0, np.cumsum(np.flip(ocp.solver_options.time_steps))))
        )
        for stage in range(2, self.planner_opts.n_nodes):
            maximum_acc = (
                    self.model.params_.maximum_deceleration_force / self.model.params_.mass
            )
            v_bound = (
                    self.planner_opts.v_max_terminal_set + times_rev[stage] * maximum_acc
            )
            v_bound = np.minimum(self.model.params_.maximum_velocity, v_bound)
            self.solver.constraints_set(
                stage, "ubx", np.array([ego_model_params.maximum_alpha, v_bound])
            )
        self._ocp = ocp
        self.last_status = []

    def warm_start(self, states_ego: np.ndarray, actions: np.ndarray):
        """
        run solver without saving data
        :return:
        """
        self.solver.reset()
        self.set_states(states_ego=states_ego, actions=actions)
        # Set reference trajectory related to set speed

        for stage in range(self.planner_opts.n_nodes):
            self.solver.set(stage, "x", states_ego)

        status = self.solver.solve()
        if status > 2:
            print(
                "WARMSTART: Acados could not find solution status = {} (status != 0)".format(
                    status
                )
            )

    def reset_solver(self):
        # with HiddenPrints():
        self.solver.load_iterate(filename=self.ini_iterate_file)

    def set_states(
            self,
            states_ego: np.ndarray,
            actions: np.ndarray = None,
            states_opp: List[np.ndarray] = None,
            t_current: float = None
    ):
        if self.is_road_circular:
            states_ego = deepcopy(states_ego)
            states_opp = deepcopy(states_opp)
            states_ego, states_opp, modulated = self.road.project_states(states_ego, states_opp)
            x_tmp = self.solver.get(0, "x")
            if modulated or np.abs(x_tmp[0] - states_ego[0]) > 10:
                for stage in range(self.planner_opts.n_nodes):
                    x_tmp = self.solver.get(stage, "x")
                    x_tmp[0] -= self.road.track_len
                    self.solver.set(stage, "x", x_tmp)

        self.states_ego = states_ego

        # get costs and references

        x_ref = np.zeros((self.n_nodes, self.nx))
        for stage in range(self.n_nodes):
            s_ref = states_ego[0] + self.time_grid[stage] * self.planner_opts.velocity_set
            x_ref[stage, :5] = np.array([s_ref, 0, 0, self.planner_opts.velocity_set, 0])

        # Set reference trajectory
        for stage in range(self.planner_opts.n_nodes - 1):
            y_ref = np.zeros((self.ny,))
            y_ref[0:self.nx] = x_ref[stage, :]
            y_ref[self.nx:self.nx + self.nu] = np.zeros((self.nu,))
            self.solver.cost_set(stage, "yref", y_ref)

        # set final reference trajectory (state only)
        y_ref = np.zeros((self.ny_e,))
        y_ref[0:self.nx] = x_ref[-1, :]
        self.solver.cost_set(self.planner_opts.n_nodes - 1, "yref", y_ref)

        # Set initial state
        self.solver.set(0, "lbx", states_ego)
        self.solver.set(0, "ubx", states_ego)

        # Set obstacle predictions
        if states_opp is not None and self.max_number_obstacles > 0:
            obstacles = []
            for state_opp in states_opp:
                if state_opp is not None:
                    obstacles.append(state_opp)
            self.set_obstacles(obstacles=obstacles)

    def solve(self) -> int:
        """
        Calls the acados solving function
        """

        self.last_status = []
        status = self.solver.solve()
        self.computation_time = self.solver.get_stats("time_tot")
        self.last_status.append(status)
        if status not in [0, 2]:
            print(
                "Warning: Acados could not find solution status = {} (status != 0)".format(
                    status
                )
            )
            print("Trying to reset acados...")
            try_out_state = self.x_full[:, 1]
            if hasattr(self, "current_ego_s"):
                try_out_state[0] -= self.current_ego_s
            self.warm_start(try_out_state, actions=None)
            status = self.solver.solve()
            self.last_status.append(status)
            if status not in [0, 2]:
                print(
                    "Error: Acados could not find solution status = {} (status != 0)".format(
                        status
                    )
                )
                # raise Exception

        self.x_full = np.empty((self.nx, self.planner_opts.n_nodes))
        self.u_full = np.empty((self.nu, self.planner_opts.n_nodes - 1))

        self.current_slacks = np.zeros((self.n_con,))
        for stage in range(1, self.planner_opts.n_nodes - 1):
            self.current_slacks = np.maximum(
                self.current_slacks, self.solver.get(stage, "sl")
            )
            self.current_slacks = np.maximum(
                self.current_slacks, self.solver.get(stage, "su")
            )

        for stage in range(self.planner_opts.n_nodes):
            self.x_full[:, stage] = self.solver.get(stage, "x")
        for stage in range(self.planner_opts.n_nodes - 1):
            self.u_full[:, stage] = self.solver.get(stage, "u")[0:2]

        return status

    def set_road(
            self,
            s_grid: np.ndarray,
            kappa_grid: np.ndarray,
            nl_grid: np.ndarray,
            nr_grid: np.ndarray,
    ):
        """
        Sets the road defined by curvature (kappa) along a grid of the center line path position
        :param s_grid: center line path grid in (m)
        :param kappa_grid: curvature along the s_grid
        :return: None
        """
        assert all(self.model.s_grid == s_grid)
        # assert not self.is_road_circular
        self.road.s_grid_ = s_grid
        self.road.kappa_grid_ = kappa_grid
        self.road.nl_grid_ = nl_grid
        self.road.nr_grid_ = nr_grid
        self.param_handler.set_par("p_kappa", kappa_grid)
        self.param_handler.set_par("p_nl", nl_grid)
        self.param_handler.set_par("p_nr", nr_grid)

    def set_obstacles(self, obstacles: List[np.ndarray]):
        """
        From the current position of the obstacles, this function sets the necessary acados parameters on each stage
        with a simple linear predictor
        :param obstacles: obstacle list, consisting of obstacles with states (s, n, alpha, v, delta). If the
        obstacle list is shorter than the defined obstacles, other obstacles will be set out or sight
        :return: nothing
        """
        # currently only two obstacles are supported
        assert len(obstacles) <= self.max_number_obstacles

        # predict obstacles linearly
        predicted_trajectories = []

        for i, opp_param in zip(
                range(self.max_number_obstacles), self.opp_model_params
        ):
            if len(obstacles[i].shape) == 2:
                if len(obstacles[i][0][:]) >= 2:
                    print("Mode not supported any more")
                    Exception()
                obstacles[i] = obstacles[i][:, 0]

            # set out of track state first
            x_out_of_track = FrenetTrajectory(
                self.out_of_track_state
            )  # this deactivates the minimal prediction

            consider_obstacle = (
                    i < len(obstacles)
                    and self.states_ego is not None
                    and not self.deactivate_obstacle_avoidance
            )
            if consider_obstacle:
                dist_ego2opp = obstacles[i][0] - self.states_ego[0]
                if dist_ego2opp > -self.planner_opts.distance_obstacle_awareness:
                    # if obstacle is considered, set corresponding prediction
                    x0_obstacle = FrenetTrajectory(obstacles[i])
                    predicted_trajectory = self.predictor.predict_linear(
                        i_opponent=i,
                        x0=x0_obstacle,
                        kappa=self.road.kappa_grid_,
                        nl=self.road.nl_grid_,
                        nr=self.road.nr_grid_,
                    )
                else:
                    predicted_trajectory = self.predictor.predict_minimal(
                        i_opponent=i,
                        x0=x_out_of_track,
                        kappa=self.road.kappa_grid_,
                        nl=self.road.nl_grid_,
                        nr=self.road.nr_grid_,
                    )
            else:
                predicted_trajectory = self.predictor.predict_numerically(
                    x0=x_out_of_track
                )

            predicted_trajectories.append(predicted_trajectory)

        # set acados parameters in stages
        obstacle_parameter_obj = self.param_handler.get_element("p_obst")
        idx_obs_start, obs_size = obstacle_parameter_obj.idx_start, obstacle_parameter_obj.size
        par_idxs_obs = np.ascontiguousarray(list(range(idx_obs_start, idx_obs_start + obs_size)))

        for stage in range(self.planner_opts.n_nodes):
            current_pos = np.empty((0,))
            for i in range(self.max_number_obstacles):
                current_pos = np.append(current_pos, predicted_trajectories[i].s[stage])
                current_pos = np.append(current_pos, predicted_trajectories[i].n[stage])
                current_pos = np.append(
                    current_pos, predicted_trajectories[i].alpha[stage]
                )  # alpha
            current_pos = np.ascontiguousarray(current_pos)
            self.solver.set_params_sparse(stage, par_idxs_obs, current_pos)

        # store predictions
        if self.planner_opts.debug_mode:
            for i in range(self.max_number_obstacles):
                predicted_trajectory = copy(predicted_trajectories[i])
                if hasattr(self, "current_ego_s"):
                    predicted_trajectory.s += self.current_ego_s
                if i < len(obstacles) and hasattr(self, "x_opposing_containers"):
                    if i == 0:
                        self.x_opp_prediction = predicted_trajectory.get_as_array()
                    self.x_opposing_containers[i] = predicted_trajectory.get_as_array()

    def get_statistics(self, fields: List[str]):
        results = []
        for field in fields:
            results.append(self.solver.get_stats(field))
        return results
