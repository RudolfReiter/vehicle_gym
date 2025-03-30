"""
Within this scenario, we test 1 devbot on a real racetrack
"""
import numpy as np
import cProfile
import pandas as pd
import scipy.signal
from matplotlib import pyplot as plt

from vehiclegym.plotting.plotters import plot_road
from vehiclegym.utils.automotive_datastructures import FrenetTrajectory
from vehiclegym.utils.helpers import json2dataclass, read_track, dataclass2json
from vehiclegym.planner.planner_reference_interfaces import InterfaceFrenet3Options, CostInterface, InterfaceFrenet3
from vehiclegym.planner.trajectory_planner_base import VehicleObstacleModel, get_time_grid
from vehiclegym.road.road import RoadOptions, Road, CircularRoad
from vehicle_models.model_kinematic import KinematicModelParameters
from vehiclegym.planner.trajectory_planner_acados_20221 import VehiclePlannerAcados20221
from vehiclegym.planner.trajectory_planner_acados_20221 import PlannerOptions
from vehiclegym.animation.animator import AnimationParameters, Animator, AnimationPlanningColorType
from vehiclegym.simulator.simulator_simple import SimulatorOptions, SimpleSimulator, PlannerEvaluator
from data import DATAPATH

if __name__ == "__main__":

    # Generate parameter data classes
    model_path = "benchmarks/model/"
    planner_path = "benchmarks/planner/"
    # Load parameters
    ego_model_params = json2dataclass(
        KinematicModelParameters, relpath=model_path, filename="devbot.json"
    )

    planner_options = json2dataclass(
        PlannerOptions, relpath=planner_path, filename="main_devbot_long_plan.json"
    )

    global_algorithms = ["FIXED_STEP", "MERIT_BACKTRACKING", "FUNNEL_L1PEN_LINESEARCH"]
    nlp_alg_colors = ["tab:blue", "tab:orange", "tab:red"]
    racetracks = ["Spielberg", "Austin", "BrandsHatch", "Budapest", "Catalunya", "Hockenheim","IMS","Melbourne"]
    n_eval = 10
    s_add = 200
    DO_PLOT = False

    res_dicts = []
    for racetrack in racetracks:
        # Create test road
        p_xy, nl, nr = read_track(name=racetrack, oversample_factor=3)
        road_options = RoadOptions()
        road = Road.from_xy(road_options=road_options, p_xy=p_xy, nl=nl, nr=nr)
        road = CircularRoad(road, s_add_neg=50, s_add_pos=500, smooth_par=(11, 3), do_plot=False)

        if DO_PLOT:
            figure = plt.figure()
            ax = figure.add_subplot(111)
            plot_road(road, fig=figure, axs=ax)

        for globalization, color in zip(global_algorithms, nlp_alg_colors):
            planner_options.globalization = globalization
            planner_options.nlp_solver_max_iter = 500
            planner_options.use_cython = False

            # Create planner
            vehicle_planner_ego = VehiclePlannerAcados20221(ego_model_params=ego_model_params,
                                                            road=road,
                                                            planner_options=planner_options,
                                                            opp_model_params=[])

            # Set parameters
            initial_state_ego_f = np.array([20, 0., 0., 0., 0])

            evaluator = PlannerEvaluator(initial_states=[initial_state_ego_f],
                                         planners=[vehicle_planner_ego],
                                         road=road,
                                         n_eval=n_eval,
                                         delta_s=s_add)

            # simulate
            # cProfile.run('simulator.simulate()', sort=1)
            res_timings, res_status4_counter, res_maximum_distances, solutions = evaluator.evaluate()

            #plot
            if DO_PLOT:
                for iter_eval, solution in enumerate(solutions):
                    x_c = road.transform_trajectory_f2c(FrenetTrajectory(solution))
                    if iter_eval == 0:
                        ax.plot(x_c.x, x_c.y, color=color,alpha=0.5,label=globalization)
                    else:
                        ax.plot(x_c.x, x_c.y, color=color, alpha=0.5)
                    ax.scatter(x_c.x, x_c.y, color=color, alpha=0.5)
                    ax.scatter(x_c.x[0], x_c.y[0], color="black", alpha=1)
                    # plot a cross at the end of the trajectory
                    ax.scatter(x_c.x[-1], x_c.y[-1], color="black",alpha=1, marker="x")

            res_dicts.append(
                {
                    "racetrack": racetrack,
                    "globalization": globalization,
                    "average_ctime": np.mean(res_timings) * 1000,
                    "acados_errors": res_status4_counter / n_eval * 100,
                    "maximum_distance": np.mean(res_maximum_distances)
                }
            )
        if DO_PLOT:
            plt.legend()
            plt.show()

    # print table
    widths = [25, 25, 15, 15, 20]
    headers = ["Race Track", "Algorithm", "Mean c. time", "acados errors", "Mean max. distance"]
    print(
        f"{headers[0]:<{widths[0]}}{headers[1]:<{widths[1]}}{headers[2]:<{widths[2]}}{headers[3]:<{widths[3]}}{headers[4]:<{widths[4]}}")

    headers = ["","","ms","%","m"]
    print(
        f"{headers[0]:<{widths[0]}}{headers[1]:<{widths[1]}}{headers[2]:<{widths[2]}}{headers[3]:<{widths[3]}}{headers[4]:<{widths[4]}}")

    print("=" * np.sum(np.array(widths)))  # Separator line

    for res_dict in res_dicts:
        row = []
        row += [res_dict["racetrack"]]
        row += [res_dict["globalization"]]
        row += [res_dict["average_ctime"]]
        row += [res_dict["acados_errors"]]
        row += [res_dict["maximum_distance"]]

        print(
            f"{row[0]:<{widths[0]}}{row[1]:<{widths[1]}}{row[2]:<{widths[2]}.3f}{row[3]:<{widths[3]}.1f}{row[4]:<{widths[4]}.3f}")

