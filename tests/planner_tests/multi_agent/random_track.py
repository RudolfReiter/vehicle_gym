"""
Within this scenario, we test 3 devbots on a randomly generated road with quite random boundaries.
"""

from copy import deepcopy
import numpy as np
import cProfile

from vehiclegym.utils.helpers import json2dataclass
from vehiclegym.planner.planner_reference_interfaces import InterfaceFrenet3Options, InterfaceFrenet3
from vehiclegym.planner.trajectory_planner_base import VehicleObstacleModel
from vehiclegym.road.road import RoadOptions, Road
from vehicle_models.model_kinematic import KinematicModelParameters
from vehiclegym.planner.trajectory_planner_acados_20221 import VehiclePlannerAcados20221
from vehiclegym.planner.trajectory_planner_acados_20221 import PlannerOptions
from vehiclegym.animation.animator import AnimationParameters, Animator, AnimationPlanningColorType
from vehiclegym.simulator.simulator_simple import SimulatorOptions, SimpleSimulator

if __name__ == "__main__":
    # general scenario parameters
    USE_RANDOM = False
    n_sim = int(250)
    idx_skip_traj = 2

    # Generate parameter data classes
    model_path = "benchmarks/model/"
    planner_path = "benchmarks/planner/"

    # Load parameters
    ego_model_params = json2dataclass(
        KinematicModelParameters, relpath=model_path, filename="devbot.json"
    )
    opp_model_params_0 = json2dataclass(
        KinematicModelParameters, relpath=model_path, filename="devbot.json"
    )
    opp_model_params_1 = json2dataclass(
        KinematicModelParameters, relpath=model_path, filename="devbot.json"
    )

    planner_options1 = json2dataclass(
        PlannerOptions, relpath=planner_path, filename="main_devbot.json"
    )
    planner_options2 = json2dataclass(
        PlannerOptions, relpath=planner_path, filename="main_devbot.json"
    )

    # Create Animation Parameter
    animation_parameter = AnimationParameters()
    animation_parameter.animation_frames_per_sec = 10
    animation_parameter.animation_speed = 1.
    animation_parameter.plot_opp_predictions = [0]
    animation_parameter.plot_safety_circles = []
    animation_parameter.plot_ego_plans = [0, 1, 2]
    animation_parameter.plot_acceleration_arrows = [0]
    animation_parameter.planning_color_type = AnimationPlanningColorType.UNI
    animation_parameter.fast_animation = True
    animation_parameter.animation_focus_ego = False

    # Create test road
    road_options = RoadOptions()
    road_options.random_road_parameters.boundary_mu = 7
    road_options.random_road_parameters.boundary_std = 5
    road_options.n_points = 4000
    road_options.random_road_parameters.maximum_kappa = 1 / 15
    road_options.random_road_parameters.sigma = 1 / 30
    road = Road(road_options)
    road.randomize(seed=3)

    # Set parameters
    simulator_options = SimulatorOptions()
    simulator_options.n_sim = int(n_sim)
    simulator_options.n_idx_skip = idx_skip_traj
    simulator_options.make_predictions_available = False

    # Create planner
    vehicle_planner_ego = VehiclePlannerAcados20221(ego_model_params=ego_model_params,
                                                    road=road,
                                                    planner_options=planner_options1,
                                                    opp_model_params=[opp_model_params_0, opp_model_params_1])

    vehicle_planner_opp_0 = VehiclePlannerAcados20221(ego_model_params=opp_model_params_0,
                                                      road=road,
                                                      planner_options=planner_options2,
                                                      opp_model_params=[ego_model_params, opp_model_params_1])

    vehicle_planner_opp_1 = VehiclePlannerAcados20221(ego_model_params=opp_model_params_1,
                                                      road=road,
                                                      planner_options=planner_options2,
                                                      opp_model_params=[opp_model_params_0, ego_model_params])

    # Set parameters
    initial_state_ego_f = np.array([1, 0., 0., 0., 0])
    initial_state_opp_0_f = np.array([40, 0, 0.0, 0., 0])
    initial_state_opp_1_f = np.array([70, 0, 0.0, 0., 0])

    simulator = SimpleSimulator(options=simulator_options,
                                initial_states=[initial_state_ego_f, initial_state_opp_0_f, initial_state_opp_1_f],
                                vehicle_parameters=[ego_model_params, opp_model_params_0, opp_model_params_1],
                                planners=[vehicle_planner_ego, vehicle_planner_opp_0, vehicle_planner_opp_1],
                                road=road)

    # simulate
    # cProfile.run('simulator.simulate()', sort=1)
    simulator.simulate()

    # print result
    simulator.print_statistics()

    animator = Animator(animation_parameter=animation_parameter)
    animator.set_data(planning_data_container=simulator.planning_containers,
                      vehicle_parameter=[ego_model_params, opp_model_params_0, opp_model_params_1],
                      planner_parameter=[planner_options1, planner_options2, planner_options2],
                      road=road,
                      statistics=simulator.statistics)
    animator.animate(save_as_movie=False)
