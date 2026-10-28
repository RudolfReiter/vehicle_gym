"""
Within this scenario, we test a devbot on a simple figure 8 racetrack
"""
import numpy as np
from matplotlib import pyplot as plt
from vehiclegym.utils.helpers import json2dataclass, read_track
from vehiclegym.planner.planner_reference_interfaces import InterfaceFrenet3Options, InterfaceFrenet3
from vehiclegym.road.road import RoadOptions, Road, CircularRoad
from vehicle_models.model_kinematic import KinematicModelParameters
from vehiclegym.planner.trajectory_planner_acados_20221 import VehiclePlannerAcados20221
from vehiclegym.planner.trajectory_planner_acados_20221 import PlannerOptions
from vehiclegym.animation.animator import AnimationParameters, Animator, AnimationPlanningColorType, AnimationPanels
from vehiclegym.simulator.simulator_simple import SimulatorOptions, SimpleSimulator
from data import DATAPATH

if __name__ == "__main__":
    # general scenario parameters
    USE_RANDOM = False
    n_sim = int(300)
    idx_skip_traj = 1

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
    opp_model_params_0.maximum_velocity = 1
    opp_model_params_1.maximum_velocity = 1

    planner_options = json2dataclass(
        PlannerOptions, relpath=planner_path, filename="main_devbot.json"
    )
    planner_options_opp = json2dataclass(
        PlannerOptions, relpath=planner_path, filename="main_devbot.json"
    )
    planner_options.use_cython = False

    # Create Animation Parameter
    animation_parameter = AnimationParameters()
    animation_parameter.animation_frames_per_sec = 10
    animation_parameter.animation_speed = 1.
    animation_parameter.plot_opp_predictions = [0, 1, 2]
    animation_parameter.plot_safety_circles = []
    animation_parameter.plot_ego_plans = [0]
    animation_parameter.plot_acceleration_arrows = [0]
    animation_parameter.planning_color_type = AnimationPlanningColorType.TIME
    animation_parameter.fast_animation = True
    animation_parameter.animation_focus_ego = False
    animation_parameter.figure_additional_size = 200
    animation_parameter.animation_pannels = AnimationPanels.CARTESIAN

    # Create test road
    N = 1000
    phi = np.linspace(0, 2 * np.pi, N)
    R = 40
    x1 = np.cos(phi) * R
    y1 = np.sin(phi) * R
    x2 = np.cos(-phi + np.pi) * R + 2 * R
    y2 = np.sin(-phi + np.pi) * R
    x = np.expand_dims(np.hstack((x1[:-1], x2)), axis=1)
    y = np.expand_dims(np.hstack((y1[:-1], y2)), axis=1)
    n_l = 5 * np.ones((2 * (N - 1) + 1,))
    n_r = 5 * np.ones((2 * (N - 1) + 1,))
    p_xy = np.hstack((x, y))
    road_tmp = Road.from_xy(road_options=RoadOptions(), p_xy=p_xy, nl=n_l, nr=n_r, smoothing_par=(51, 3), do_plot=True)
    road = CircularRoad(road_tmp, s_add_neg=50, s_add_pos=200)
    plt.plot(road_tmp.s_grid_, road_tmp.kappa_grid_)
    plt.plot(road.s_grid_, road.kappa_grid_)
    plt.show()
    # road.kappa_grid_ = np.ones_like(road.kappa_grid_) * road.kappa_grid_[10]

    # Set parameters
    simulator_options = SimulatorOptions()
    simulator_options.n_sim = int(n_sim)
    simulator_options.n_idx_skip = idx_skip_traj
    simulator_options.make_predictions_available = False

    # Create planner
    vehicle_planner_ego = VehiclePlannerAcados20221(ego_model_params=ego_model_params,
                                                    road=road,
                                                    planner_options=planner_options,
                                                    opp_model_params=[opp_model_params_0, opp_model_params_1])
    vehicle_planner_opp_0 = VehiclePlannerAcados20221(ego_model_params=opp_model_params_0,
                                                      road=road,
                                                      planner_options=planner_options_opp,
                                                      opp_model_params=[ego_model_params, opp_model_params_1])

    vehicle_planner_opp_1 = VehiclePlannerAcados20221(ego_model_params=opp_model_params_1,
                                                      road=road,
                                                      planner_options=planner_options_opp,
                                                      opp_model_params=[opp_model_params_0, ego_model_params])

    # Set parameters
    initial_state_ego_f = np.array([80, 0., 0., 0., 0])
    initial_state_opp_0_f = np.array([10, 0, 0.0, 0., 0])
    initial_state_opp_1_f = np.array([20, 0, 0.0, 0., 0])

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
                      planner_parameter=[planner_options, planner_options_opp, planner_options_opp],
                      road=road,
                      statistics=simulator.statistics)
    animator.animate(save_as_movie=False)
