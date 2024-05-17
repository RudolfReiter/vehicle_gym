# This simulator assumes perfect tracking of a feasible planned trajectory
from abc import ABC
from copy import copy, deepcopy
from dataclasses import dataclass
import time
from typing import List, Tuple
import numpy as np
from vehicle_models.model_kinematic_frenet import FrenetModel

from vehicle_models.model_kinematic import KinematicModelParameters, KinematicVehicleModel
from vehiclegym.utils.automotive_datastructures import FrenetTrajectory
from vehiclegym.utils.evalutation_tools import get_minimum_distance_multi, check_collision_multi
from vehiclegym.simulator.simulation_datastructures import KinodynamicSimulator
from vehiclegym.road.road import Road
from vehiclegym.simulator.simulation_datastructures import RaceSimulatorOptions, SimulationStatistics
from vehiclegym.simulator.simulator_simple import BaseSimulator
from vehiclegym.planner.trajectory_planner_acados_iterative import VehiclePlannerAcados2022Iter
from vehiclegym.planner.trajectory_planner_base import PlanningDataContainer


class RaceSimulator(BaseSimulator):
    def __init__(self,
                 options: RaceSimulatorOptions,
                 initial_states: List[np.ndarray],
                 vehicle_parameters: List[KinematicModelParameters],
                 planners: List[VehiclePlannerAcados2022Iter],
                 road: Road,
                 models: List[KinematicVehicleModel] = None):

        tds = [planner.planner_opts.time_disc for planner in planners]
        assert len(initial_states) == len(vehicle_parameters) == len(planners)
        assert len(set(tds)) == 1  # are all elements (tds) equal?
        self.vehicle_parameters = vehicle_parameters
        if models is None:
            models = []
            for planner, vehicle_parameter in zip(planners, vehicle_parameters):
                models.append(FrenetModel(s_grid=road.s_grid_, p_kappa=road.kappa_grid_, params=vehicle_parameter,
                                          p_nl=road.nl_grid_, p_nr=road.nr_grid_))

        super().__init__(options, initial_states, vehicle_parameters, road, models, tds[0])

        self.planners = planners
        self.full_trajectories = None
        self.road = road
        self.solver_times = None
        self.full_planner_time = []
        self.solver_wrapper_time = []
        self.all_statii = []
        self.max_distance_reached = []
        self.statistics = SimulationStatistics(n_vehicles=self.n_vehicles, n_sim=options.n_sim,
                                               delta_t=self.td * options.n_idx_skip)
        self.current_states = initial_states

        self.planning_containers = []
        for i in range(self.n_vehicles):
            self.planning_containers.append(PlanningDataContainer())

        for i, (planner, initial_state) in enumerate(zip(planners, initial_states)):
            action = self.default_actions[i]
            planner.initialize_iter(states_ego=initial_state, actions=action, s_grid=road.s_grid_,
                                    kappa_grid=road.kappa_grid_, nl_grid=road.nl_grid_, nr_grid=road.nr_grid_)

    def reset(self, initial_states: List[np.ndarray], road: Road):
        self.statistics = SimulationStatistics(n_vehicles=self.n_vehicles, n_sim=self.options.n_sim,
                                               delta_t=self.td * self.options.n_idx_skip)
        # set initial states to zero
        for sim in self.kino_simulators:
            sim.reset()

        # reinitialize models in simulator due to kappa
        models = []
        for planner, vehicle_parameter in zip(self.planners, self.vehicle_parameters):
            models.append(
                FrenetModel(s_grid=road.s_grid_, p_kappa=road.kappa_grid_, params=vehicle_parameter, p_nl=road.nl_grid_,
                            p_nr=road.nr_grid_))
        self.initialize_simulators(models=models, initial_states=initial_states, td=self.td)

        self.current_states = initial_states
        self.planning_containers = []
        self.full_planner_time = []
        self.solver_wrapper_time = []
        self.all_statii = []
        self.max_distance_reached = []
        self.solver_times = None

        # set road in planners
        for i in range(self.n_vehicles):
            self.planning_containers.append(PlanningDataContainer())

        for i, (planner, initial_state) in enumerate(zip(self.planners, initial_states)):
            action = self.default_actions[i]
            planner.initialize_iter(states_ego=initial_state, actions=action, s_grid=road.s_grid_,
                                    kappa_grid=road.kappa_grid_, nl_grid=road.nl_grid_, nr_grid=road.nr_grid_)

    def simulate(self):
        self.solver_times = np.empty((self.n_vehicles, self.options.n_sim))
        ego_first = False
        self.time2lead = -1

        for i in range(self.options.n_sim):
            # set simulation time
            t_sim = i * self.options.n_idx_skip * self.td

            # Solve and time vehicles
            planning_datas = []
            timings = []
            new_states = []
            full_trajectories = []

            # solve all planners for current setting
            for i_planner, [planner, container, kino_simulator] in enumerate(zip(self.planners,
                                                                                 self.planning_containers,
                                                                                 self.kino_simulators)):
                t_start = time.time()
                # set initial states
                local_states = deepcopy(self.current_states)
                # for i_state, state in enumerate(local_states):
                # c_state = self.road.transform_trajectory_f2c(FrenetTrajectory(state))
                # f_state = self.road.transform_trajectory_c2f(c_state, initial_guess_s=state[0])
                # f_state = self.road.transform_trajectory_c2f_fast(c_state)
                # local_states[i_state] = f_state.get_as_array().flatten()
                other_states = local_states[:i_planner] + local_states[i_planner + 1:]

                # set states and actions, main solver call
                planner.solve_iter(states_ego=local_states[i_planner],
                                   states_opp=other_states,
                                   actions=self.default_actions[i_planner],
                                   s_grid=self.road.s_grid_, kappa_grid=self.road.kappa_grid_,
                                   nl_grid=self.road.nl_grid_, nr_grid=self.road.nr_grid_)

                self.solver_wrapper_time.append(time.time() - t_start)
                self.all_statii.extend(planner.last_status)
                self.full_planner_time.append(time.time() - t_start)

                # get timings
                time_iter = planner.solver.get_stats("time_tot")
                timings.append(time_iter)

                # get next state
                planning_datas.append(planner.get_formatted_solution(t0=t_sim))

                # Save formatted solution to container with all data
                container.add(planning_datas[-1])

                # simulate real vehicle
                new_state = None
                for j in range(self.options.n_idx_skip):
                    new_state = kino_simulator.step(u_control=planner.u_full[:, j])

                # get state
                new_states.append(new_state)
                x_full = np.zeros_like(planning_datas[-1].x)
                x_full[:, :-self.options.n_idx_skip] = planning_datas[-1].x[:, self.options.n_idx_skip:]
                x_full[:, -self.options.n_idx_skip:] = np.expand_dims(x_full[:, -self.options.n_idx_skip - 1], 1)
                full_trajectories.append(x_full)

            first_idx = np.argmax(np.array([state[0] for state in new_states]))
            if not ego_first and first_idx == 0:
                self.time2lead = copy(t_sim)
                ego_first = True

            # we need a second loop here because all opponents move simultaneously
            # after all trajectories were computed we perform some checks
            self.current_states = new_states
            self.full_trajectories = full_trajectories
            for i_planner in range(len(self.planners)):

                # check for collisions
                collision_leader_follower = None
                if self.options.check_crashes:
                    other_states = self.current_states[:i_planner] + self.current_states[i_planner + 1:]
                    other_parameters = self.vehicle_parameters[:i_planner] + self.vehicle_parameters[i_planner + 1:]
                    is_collision, idx_opp_rel, is_ego_lead = check_collision_multi(
                        ego_state=self.current_states[i_planner],
                        ego_parameter=self.vehicle_parameters[i_planner],
                        opp_states=other_states,
                        opp_parameters=other_parameters,
                        road=self.road)
                    if is_collision:
                        idx_ego = i_planner
                        idx_opp = idx_opp_rel[0] + 1 * (i_planner <= idx_opp_rel[0])
                        if is_ego_lead:
                            collision_leader_follower = (idx_ego, idx_opp)
                        else:
                            collision_leader_follower = (idx_opp, idx_ego)

                # Save timings
                if self.options.get_statistics:
                    [t_sim, t_qp, t_glob, t_reg, t_lin, qp_iter] = self.planners[i_planner].get_statistics(["time_sim",
                                                                                                            "time_qp",
                                                                                                            "time_glob",
                                                                                                            "time_reg",
                                                                                                            "time_lin",
                                                                                                            "qp_iter"])
                    self.statistics.add(computation_time=timings[i_planner],
                                        collision=collision_leader_follower,
                                        time_sim=t_sim,
                                        time_qp=t_qp,
                                        time_glob=t_glob,
                                        time_reg=t_reg,
                                        time_lin=t_lin,
                                        qp_iter=qp_iter)
            self.max_distance_reached.append(self.current_states[0][0])

    def get_statistics(self):
        t_acados_average = np.mean(self.statistics.solver_times)
        t_plan_mean = np.mean(np.array(self.full_planner_time))
        t_wrap_mean = np.mean(np.array(self.solver_wrapper_time))
        sum_collisions = np.sum(self.statistics.collisions_lead) + \
                         np.sum(self.statistics.collisions_follow)
        time2lead = self.time2lead
        max_distance = np.mean(np.array(self.max_distance_reached))

        return [t_acados_average, t_wrap_mean, t_plan_mean, sum_collisions, time2lead, max_distance]

    def get_statii(self):
        return [self.all_statii.count(0),
                self.all_statii.count(1),
                self.all_statii.count(2),
                self.all_statii.count(3),
                self.all_statii.count(4)]

    def print_statistics(self):
        """ Print statistics of the simulation"""

        t_average = np.mean(self.statistics.solver_times) * 1000.
        t_max = np.max(self.statistics.solver_times) * 1000.
        t_min = np.min(self.statistics.solver_times) * 1000.

        sum_collisions_leader = np.sum(self.statistics.collisions_lead, axis=1)
        sum_collisions_follow = np.sum(self.statistics.collisions_follow, axis=1)
        min_distances = np.min(self.statistics.min_distances, axis=1)

        t_plan_mean = np.mean(np.array(self.full_planner_time)) * 1000
        t_wrap_mean = np.mean(np.array(self.solver_wrapper_time)) * 1000

        print("Statistics of simulation run:")
        print("-----------------------------")
        print("Computation time acados (ms) [ave/min/max]: {:4.2f}/ {:4.2f}/ {:4.2f}".format(t_min, t_average, t_max))
        print("Time to lead: {:4.2f} s".format(self.time2lead))
        print("Full planner time: {:4.2f} ms".format(t_plan_mean))
        print("Full solver wrapper time: {:4.2f} ms".format(t_wrap_mean))
        for i in range(len(sum_collisions_leader)):
            print("Sum of lead collisions vehicle " + str(i) + ": {}".format(sum_collisions_leader[i]))
        for i in range(len(sum_collisions_follow)):
            print("Sum of follow collisions vehicle " + str(i) + ": {}".format(sum_collisions_follow[i]))
        for i in range(len(sum_collisions_leader)):
            print("Minimum opponent distance of vehicle " + str(i) + ": {}".format(min_distances[i]))
