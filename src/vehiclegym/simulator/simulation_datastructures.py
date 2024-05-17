import numpy as np

from vehiclegym.utils.evalutation_tools import check_collision_multi
from vehiclegym.planner.trajectory_planner_base import (PlannerOptions,
                                                        PlanningDataContainer)
import casadi as cs
from vehicle_models.model_kinematic import KinematicVehicleModel
from copy import copy
from dataclasses import dataclass
from vehiclegym.road.road import RoadOptions, Road, FrozenClass
from vehicle_models.model_kinematic import KinematicModelParameters
from vehicle_models.model_kinematic_frenet import FrenetModel
from typing import List, Tuple, TYPE_CHECKING
import enum

if TYPE_CHECKING:
    from vehiclegym.animation.animator import AnimationParameters


@dataclass
class SimulatorOptions(FrozenClass):
    n_sim: int = 100  # number of simulation steps
    n_idx_skip: int = 1  # number of planned indexes skipped
    check_distances: bool = True
    check_crashes: bool = True
    get_statistics: bool = True
    warm_start: bool = True
    make_predictions_available: bool = False
    sim_cov_matrix: np.ndarray = None
    check_ellipsoidal: bool = False


@dataclass
class RaceSimulatorOptions(SimulatorOptions):
    n_road_segment: int = 1000
    n_road_segment_pre: int = 200


class SimulationStatistics:
    def __init__(self, n_vehicles: int, n_sim: int, delta_t: float):
        self.solver_times = np.empty((n_vehicles, n_sim))
        self.collisions_lead = np.zeros((n_vehicles, n_sim))
        self.collisions_follow = np.zeros((n_vehicles, n_sim))
        self.collision_static_obs = np.zeros((n_vehicles, n_sim))
        self.min_distances = np.empty((n_vehicles, n_sim))
        self.time_sims = np.empty((n_vehicles, n_sim))
        self.time_qps = np.empty((n_vehicles, n_sim))
        self.time_glob = np.empty((n_vehicles, n_sim))
        self.time_reg = np.empty((n_vehicles, n_sim))
        self.time_lin = np.empty((n_vehicles, n_sim))
        self.qp_iters = np.empty((n_vehicles, n_sim))

        self.current_vehicle = 0
        self.current_step = 0
        self.delta_t = delta_t

        self.time_steps = []
        self.n_vehicles = n_vehicles

    def add(self, computation_time: float,
            collision: Tuple[int, int],
            time_sim: float,
            time_qp: float,
            time_glob: float,
            time_reg: float,
            time_lin: float,
            qp_iter: int,
            col_static: bool= None,
            i_vehicle=None, i_step=None):
        if self.time_steps.__len__() == 0:
            self.time_steps.append(0)
        else:
            self.time_steps.append(self.time_steps[-1] + self.delta_t)
        if i_vehicle is None:
            i_vehicle = self.current_vehicle
        if i_step is None:
            i_step = self.current_step
        self.solver_times[i_vehicle, i_step] = computation_time
        self.time_sims[i_vehicle, i_step] = time_sim
        self.time_qps[i_vehicle, i_step] = time_qp
        self.time_reg[i_vehicle, i_step] = time_reg
        self.time_lin[i_vehicle, i_step] = time_lin
        self.time_glob[i_vehicle, i_step] = time_glob
        self.collision_static_obs[i_vehicle, i_step] = col_static
        self.qp_iters[i_vehicle, i_step] = qp_iter[-1]
        if collision is not None:
            self.collisions_lead[collision[0], i_step] = 1
            self.collisions_follow[collision[1], i_step] = 1

        self.current_vehicle += 1
        self.current_vehicle = np.mod(self.current_vehicle, self.n_vehicles)

        if self.current_vehicle == 0:
            self.current_step += 1


class InfeasibilityType(enum.Enum):
    VEHICLE_COLLISION_EGO_LEAD = enum.auto()
    VEHICLE_COLLISION_OPP_LEAD = enum.auto()
    BOUNDARY_COLLISION = enum.auto()
    SOFT_BOUNDARY_COLLISION = enum.auto()
    LATERAL_ACCELERATION = enum.auto()
    MAXIMUM_VELOCITY = enum.auto()
    SOFT_MAXIMUM_VELOCITY = enum.auto()
    MAXIMUM_STEERING_ANGLE = enum.auto()
    OUT_OF_ROAD_RANGE = enum.auto()
    MAXIMUM_HEADING_ANGLE = enum.auto()
    NONE = enum.auto()

    def is_soft(self):
        soft_enums = [InfeasibilityType.SOFT_BOUNDARY_COLLISION,
                      InfeasibilityType.SOFT_MAXIMUM_VELOCITY]
        return self in soft_enums


@dataclass
class RacingSimulationData:
    ego_planning_data: PlanningDataContainer = None
    opp_planning_data: PlanningDataContainer = None
    ego_model_parameters: KinematicModelParameters = None
    opp_model_parameters: KinematicModelParameters = None
    ego_planner_options: PlannerOptions = None
    opp_planner_options: PlannerOptions = None
    road_parameters: RoadOptions = None
    idx_skipped_simulation: int = 0

    def prepare_for_animation(self, animation_parameter: "AnimationParameters"):
        self.ego_planning_data.flatten_data()
        resample_delta_t = 1 / animation_parameter.animation_frames_per_sec * animation_parameter.animation_speed
        self.ego_planning_data.resample(self.ego_planner_options.time_disc, resample_delta_t)
        self.opp_planning_data.flatten_data()
        self.opp_planning_data.resample(self.ego_planner_options.time_disc, resample_delta_t)


class KinodynamicSimulator:
    def __init__(self, model: KinematicVehicleModel,
                 time_disc: float,
                 x_states_0: np.ndarray,
                 track_len: float = None,
                 cov_matrix: np.ndarray = None):

        self.f_dyn = model.f_ode
        self.time_disc = time_disc
        self.model = model
        self.track_len = track_len
        self.cov_matrix = cov_matrix

        k1 = self.f_dyn(model.x_states, model.u_controls)
        k2 = self.f_dyn(model.x_states + self.time_disc / 2 * k1, model.u_controls)
        k3 = self.f_dyn(model.x_states + self.time_disc / 2 * k2, model.u_controls)
        k4 = self.f_dyn(model.x_states + self.time_disc * k3, model.u_controls)
        x_next = model.x_states + self.time_disc / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        self.f_integrator = cs.Function('F_int', [model.x_states, model.u_controls], [x_next])
        self.x0 = x_states_0
        self.x_state = x_states_0

        self.four_state_model = False
        if model.n_x == 4:
            self.four_state_model = True

    def step(self, u_control: np.ndarray, x_state=None) -> np.ndarray:
        if x_state is None:
            x_state = self.x_state
        self.x_state = self.f_integrator(x_state, u_control).full()[:, 0]
        # Hackmack, because in one particular model we have the steering angle delta as a state
        if self.four_state_model:
            self.x_state[4] = u_control[1]
        self.x_state[4] = np.clip(self.x_state[4], -self.model.params_.maximum_steering_angle,
                                  self.model.params_.maximum_steering_angle)
        self.x_state = copy(self.x_state)
        if self.track_len is not None:
            self.x_state[0] = np.mod(self.x_state[0], self.track_len)

        if self.cov_matrix is not None:
            noise = np.random.multivariate_normal(np.zeros((self.cov_matrix.shape[0],)), self.cov_matrix)
            self.x_state += noise
        return self.x_state

    def reset(self, x0: np.ndarray = None):
        self.x_state = self.x0 if x0 is None else x0


class FeasibilityChecker:
    def __init__(self, ego_model: FrenetModel,
                 road: Road,
                 opp_models_params: List[KinematicModelParameters] = None,
                 epsilon: float = 1e-6,
                 soft_road_bounds: bool = False,
                 ellipsoidal_obstacles: bool = False):
        self.ego_model = ego_model
        self.ego_model_params = ego_model.params_
        self.opp_models_params = opp_models_params
        self.road = road
        self.epsilon = epsilon
        self.soft_road_bounds = soft_road_bounds
        self.ellipsoidal_obstacles = ellipsoidal_obstacles

    def check_feasibility(self, x_state_ego: np.ndarray, x_states_opp: List[np.ndarray] = None) -> \
            Tuple[bool, InfeasibilityType]:

        def soft_hard_constraint(x, soft_min_x=-np.inf, soft_max_x=np.inf, hard_min_x=-np.inf, hard_max_x=np.inf,
                                 infeasibility_type=None, soft_infeasiblity_type=None):
            assert hard_min_x <= soft_min_x <= soft_max_x <= hard_max_x

            if soft_min_x - self.epsilon < x < soft_max_x + self.epsilon:
                return None
            elif hard_min_x - self.epsilon < x < hard_max_x + self.epsilon and self.soft_road_bounds:
                return soft_infeasiblity_type
            # import pdb; pdb.set_trace()
            return infeasibility_type

        if x_states_opp is not None:
            if len(x_states_opp) > 0:
                is_collision, idx_opp_rel, is_ego_lead = check_collision_multi(
                    ego_state=x_state_ego,
                    ego_parameter=self.ego_model_params,
                    opp_states=x_states_opp,
                    opp_parameters=self.opp_models_params,
                    road=self.road,
                    check_ellipsoidal=self.ellipsoidal_obstacles)
                if is_collision:
                    if is_ego_lead:
                        infeasibility_type = InfeasibilityType.VEHICLE_COLLISION_EGO_LEAD
                    else:
                        infeasibility_type = InfeasibilityType.VEHICLE_COLLISION_OPP_LEAD
                    return False, infeasibility_type

        # min_road_n = -(self.road.road_options_.road_width / 2 - self.ego_model_params.safety_radius)
        # if min_road_n > x_state_ego[1] - self.epsilon:
        #    return False, InfeasibilityType.BOUNDARY_COLLISION

        # max_road_n = self.road.road_options_.road_width / 2 - self.ego_model_params.safety_radius
        # if max_road_n < x_state_ego[1] - self.epsilon:
        #    return False, InfeasibilityType.BOUNDARY_COLLISION

        if self.ego_model_params.maximum_steering_angle < x_state_ego[4] - self.epsilon:
            return False, InfeasibilityType.MAXIMUM_STEERING_ANGLE

        if np.abs(self.ego_model.f_a_lat(x_state_ego)) - self.epsilon > self.ego_model.params_.maximum_lateral_acc:
            return False, InfeasibilityType.LATERAL_ACCELERATION

        if x_state_ego[0] < self.road.s_grid_[0] or x_state_ego[0] > self.road.s_grid_[-1]:
            return False, InfeasibilityType.OUT_OF_ROAD_RANGE

        if x_state_ego[2] < -self.ego_model.params_.maximum_alpha or \
                x_state_ego[2] > self.ego_model.params_.maximum_alpha:
            return False, InfeasibilityType.MAXIMUM_HEADING_ANGLE

        # TODO: This should be strict here
        soft_max_road_n = self.road.spline_s2nl_(x_state_ego[0]) - self.ego_model_params.safety_radius_boundary
        soft_min_road_n = -(self.road.spline_s2nr_(x_state_ego[0]) - self.ego_model_params.safety_radius_boundary)
        # hard_min_road_n = soft_min_road_n * 1.5
        # hard_max_road_n = soft_max_road_n * 1.5
        hard_min_road_n = soft_min_road_n
        hard_max_road_n = soft_max_road_n
        bound_check = soft_hard_constraint(x_state_ego[1], soft_min_road_n, soft_max_road_n, hard_min_road_n,
                                           hard_max_road_n, InfeasibilityType.BOUNDARY_COLLISION,
                                           InfeasibilityType.SOFT_BOUNDARY_COLLISION)
        if bound_check is not None:
            return False, bound_check

        soft_max_v = self.ego_model_params.maximum_velocity
        hard_max_v = self.ego_model_params.maximum_velocity * 1.1
        vel_check = soft_hard_constraint(x_state_ego[3], soft_max_x=soft_max_v, hard_max_x=hard_max_v,
                                         infeasibility_type=InfeasibilityType.BOUNDARY_COLLISION,
                                         soft_infeasiblity_type=InfeasibilityType.SOFT_BOUNDARY_COLLISION)
        if vel_check is not None:
            return False, vel_check

        return True, InfeasibilityType.NONE
