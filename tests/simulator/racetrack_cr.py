"""
Within this scenario, we test 1 devbot on a real racetrack
"""
import numpy as np
import cProfile
import pandas as pd
import scipy.signal
from vehiclegym.utils.helpers import json2dataclass, read_track, dataclass2json
from vehiclegym.planner.planner_reference_interfaces import InterfaceFrenet3Options, CostInterface, InterfaceFrenet3
from vehiclegym.planner.trajectory_planner_base import VehicleObstacleModel, get_time_grid
from vehiclegym.road.road import RoadOptions, Road, CircularRoad
from vehicle_models.model_kinematic import KinematicModelParameters
from vehiclegym.planner.trajectory_planner_acados_20221 import VehiclePlannerAcados20221
from vehiclegym.planner.trajectory_planner_acados_20221 import PlannerOptions
from vehiclegym.animation.animator import AnimationParameters, Animator, AnimationPlanningColorType
from vehiclegym.simulator.simulator_cr import SimulatorOptions, CrSimulator, get_par_from_cmrd, VehicleModelType
from vehiclemodels.parameters_vehicle2 import parameters_vehicle2
from vehiclemodels.parameters_vehicle1 import parameters_vehicle1
from data import DATAPATH

if __name__ == "__main__":
    # general scenario parameters
    USE_RANDOM = False
    n_sim = int(400)
    idx_skip_traj = 1

    # Generate parameter data classes
    model_path = "benchmarks/model/"
    planner_path = "benchmarks/planner/"
    # Load parameters
    ego_model_params_cr = parameters_vehicle1()
    ego_model_params = get_par_from_cmrd(ego_model_params_cr, fac_acc_lon=0.3, fac_dec_lon=0.3, fac_a_lat=0.2)
    ego_model_params = get_par_from_cmrd(ego_model_params_cr)

    planner_options = json2dataclass(
        PlannerOptions, relpath=planner_path, filename="main_devbot.json"
    )
    planner_options.use_cython = False

    # Create Animation Parameter
    animation_parameter = AnimationParameters()
    animation_parameter.animation_frames_per_sec = 10
    animation_parameter.animation_speed = 1.
    animation_parameter.plot_opp_predictions = [0]
    animation_parameter.plot_safety_circles = []
    animation_parameter.plot_ego_plans = [0]
    animation_parameter.plot_acceleration_arrows = [0]
    animation_parameter.planning_color_type = AnimationPlanningColorType.TIME
    animation_parameter.fast_animation = True
    animation_parameter.animation_focus_ego = False
    animation_parameter.figure_additional_size = 100

    # Create test road
    CREATE_RANDOM = True
    if not CREATE_RANDOM:
        # create race track
        p_xy, nl, nr = read_track(name="Monza", oversample_factor=3)
        road_options = RoadOptions()
        road = Road.from_xy(road_options=road_options, p_xy=p_xy, nl=nl, nr=nr)
        road = CircularRoad(road, s_add_neg=50, s_add_pos=500, smooth_par=(11, 3), do_plot=False)
    else:
        # Create test road
        road_options = RoadOptions()
        road_options.random_road_parameters.boundary_mu = 7
        road_options.random_road_parameters.boundary_std = 5
        road_options.n_points = 4000
        road_options.random_road_parameters.maximum_kappa = 1 / 15
        road_options.random_road_parameters.sigma = 1 / 30
        road = Road(road_options)
        road.randomize(seed=-1)

    # Set parameters
    simulator_options = SimulatorOptions()
    simulator_options.n_sim = int(n_sim)
    simulator_options.n_idx_skip = idx_skip_traj
    simulator_options.make_predictions_available = False

    # Create planner
    vehicle_planner_ego = VehiclePlannerAcados20221(ego_model_params=ego_model_params,
                                                    road=road,
                                                    planner_options=planner_options,
                                                    opp_model_params=[])

    # Set parameters
    initial_state_ego_f = np.array([20, 0., 0., 10., 0])

    simulator = CrSimulator(options=simulator_options,
                            road=road,
                            initial_states_f=[initial_state_ego_f],
                            model_type=VehicleModelType.KS,
                            vehicle_par_cr=[ego_model_params_cr],
                            planners=[vehicle_planner_ego])

    # simulate
    # cProfile.run('simulator.simulate()', sort=1)
    simulator.simulate()

    # print result
    simulator.print_statistics()

    animator = Animator(animation_parameter=animation_parameter)
    animator.set_data(planning_data_container=simulator.planning_containers,
                      vehicle_parameter=[ego_model_params],
                      planner_parameter=[planner_options],
                      road=road,
                      statistics=simulator.statistics)
    animator.animate(save_as_movie=False)
