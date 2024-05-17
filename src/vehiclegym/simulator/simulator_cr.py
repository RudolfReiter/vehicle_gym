# This simulator assumes perfect tracking of a feasible planned trajectory
import enum
from abc import ABC
from copy import copy, deepcopy
from dataclasses import dataclass
import time
from typing import List, Tuple
import numpy as np
from omegaconf import DictConfig
from scipy.integrate import odeint

from vehicle_models.model_kinematic_frenet import FrenetModel

from vehicle_models.model_kinematic import KinematicModelParameters, KinematicVehicleModel
from vehiclegym.utils.automotive_datastructures import FrenetTrajectory, CartesianTrajectory
from vehiclegym.utils.evalutation_tools import get_minimum_distance_multi, check_collision_multi
from vehiclegym.simulator.simulation_datastructures import KinodynamicSimulator
from vehiclegym.road.road import Road
from vehiclegym.simulator.simulation_datastructures import SimulatorOptions, SimulationStatistics
from vehiclegym.planner.trajectory_planner_acados_20221 import VehiclePlannerAcados20221
from vehiclegym.planner.trajectory_planner_base import PlanningDataContainer
from vehiclemodels.init_mb import init_mb
from vehiclemodels.init_st import init_st
from vehiclemodels.init_ks import init_ks
from vehiclemodels.init_std import init_std
from vehiclemodels.vehicle_dynamics_mb import vehicle_dynamics_mb
from vehiclemodels.vehicle_dynamics_st import vehicle_dynamics_st
from vehiclemodels.vehicle_dynamics_ks import vehicle_dynamics_ks
from vehiclemodels.vehicle_dynamics_std import vehicle_dynamics_std


class VehicleModelType(enum.Enum):
    """ Defines which vehicle model should be used """
    KS = enum.auto()  # kinematic single track 5 states
    DS = enum.auto()  # dynamic single track 6 states, linearized Pacejka
    DSD = enum.auto()  # dynamic single track drift 6 states, nonlinear Pacejka
    MB = enum.auto()  # multi body double track 29 states


def get_par_from_cmrd(params_cmrd, fac_acc_lon: float = 1, fac_dec_lon: float = 1,fac_a_lat: float = 1) -> KinematicModelParameters:
    params = KinematicModelParameters()
    params.mass = params_cmrd.m
    params.length_rear = params_cmrd.a  # can not find lf, lr
    params.length_front = params_cmrd.b # can not find lf, lr
    params.chassis_length = params_cmrd.l
    params.chassis_width = params_cmrd.w
    params.maximum_velocity = params_cmrd.longitudinal.v_max
    params.maximum_steering_rate = params_cmrd.steering.v_max
    params.maximum_steering_angle = params_cmrd.steering.max
    params.maximum_lateral_acc = params_cmrd.longitudinal.a_max * fac_a_lat
    params.maximum_acceleration_force = params_cmrd.longitudinal.a_max * params.mass * fac_acc_lon
    # cmrd computes the acceleration more accurately
    params.maximum_deceleration_force = params_cmrd.longitudinal.a_max * params.mass * fac_dec_lon
    # cmrd computes the acceleration more accurately
    return params

import collections

def flatten(l):
    for el in l:
        if isinstance(el, collections.abc.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el

class CrSimulator:
    def __init__(self,
                 options: SimulatorOptions,
                 road: Road,
                 model_type: VehicleModelType,
                 vehicle_par_cr: List[DictConfig],
                 initial_states_f: List[np.ndarray],
                 planners: List[VehiclePlannerAcados20221],
                 ):

        tds = [planner.planner_opts.time_disc for planner in planners]
        assert len(initial_states_f) == len(vehicle_par_cr) == len(planners)
        assert len(set(tds)) == 1  # are all elements (tds) equal?
        self.vehicle_parameters = vehicle_par_cr
        self.n_vehicles = len(initial_states_f)
        self.model_type= model_type
        self.u_acc_pre = 0

        # choose functions to simulate
        init_funs = []
        vehicle_dynamic_funs = []
        for par in vehicle_par_cr:
            if model_type is VehicleModelType.KS:
                init_fun = init_ks
                vehicle_dynamics_fun = vehicle_dynamics_ks
            elif model_type is VehicleModelType.DS:
                init_fun = init_st
                vehicle_dynamics_fun = vehicle_dynamics_st
            elif model_type is VehicleModelType.DSD:
                init_fun = lambda x: init_std(x, par)
                vehicle_dynamics_fun = vehicle_dynamics_std
            elif model_type is VehicleModelType.MB:
                init_fun = lambda x: init_mb(x, par)
                vehicle_dynamics_fun = vehicle_dynamics_mb
            else:
                raise NotImplementedError
            vehicle_dynamic_funs.append(vehicle_dynamics_fun)
            init_funs.append(init_fun)
        self.init_funs_ = init_funs

        # transform initial states and append simulation functions
        initial_states_c, self.current_states_c, self.funcs_sim, self.funcs_sim_ls = [], [], [], []
        for state, p, init_fun in zip(initial_states_f, vehicle_par_cr, init_funs):
            state = FrenetTrajectory(state)
            initial_states_c.append(road.transform_trajectory_f2c(state))
            tmp = init_fun([initial_states_c[-1].x,
                               initial_states_c[-1].y,
                               initial_states_c[-1].delta,
                               initial_states_c[-1].v,
                               initial_states_c[-1].phi,
                               0, 0])
            #state_c = np.array(flatten(tmp))[:,0]
            state_c = np.array(list(flatten(tmp)))
            self.current_states_c.append(state_c)  # p, delta0, vel0, Psi0, dotPsi0, beta0]))

            def func_sim(x, t, u):
                f = vehicle_dynamics_fun(x, u, p)
                return f

            self.funcs_sim.append(func_sim)

            def func_sim_ls(x, t, u):
                f = vehicle_dynamics_ks(x, u, p)
                return f

            self.funcs_sim_ls.append(func_sim_ls)

        self.options = options
        self.td = tds[0]
        self.road = road
        self.default_actions = [None] * self.n_vehicles
        track_len = None
        if hasattr(road, "track_len"):
            track_len = road.track_len
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

        self.planning_containers = []
        for i in range(self.n_vehicles):
            self.planning_containers.append(PlanningDataContainer())

        for i, (planner, initial_state) in enumerate(zip(planners, initial_states_f)):
            if self.options.warm_start:
                planner.warm_start(states_ego=initial_state, actions=None)
            else:
                planner.solver.reset()
            other_states = initial_states_f[:i] + initial_states_f[i + 1:]
            planner.set_states(states_ego=initial_state, states_opp=other_states, t_current=0)

    def set_road(self, road: Road):
        for planner in self.planners:
            planner.set_road(s_grid=road.s_grid_, kappa_grid=road.kappa_grid_, nl_grid=road.nl_grid_,
                             nr_grid=road.nr_grid_)

    def simulate(self):
        self.solver_times = np.empty((self.n_vehicles, self.options.n_sim))
        ego_first = False
        self.time2lead = -1
        planner_states_f = []
        for i in range(self.options.n_sim):
            # set simulation time
            t_sim = i * self.options.n_idx_skip * self.td

            # Solve and time vehicles
            planning_datas = []
            timings = []
            full_trajectories = []
            initial_s = [copy(state[0]) for state in planner_states_f]
            if initial_s.__len__() == 0:
                initial_s = np.array([0.] * self.n_vehicles)

            # transform states to frenet state for planner
            planner_states_f = []
            for state, ini_s, params in zip(self.current_states_c, initial_s,self.vehicle_parameters):
                planner_state_c = CartesianTrajectory(np.array([state[0],
                                                                state[1],
                                                                state[4],
                                                                state[3],
                                                                state[2], ]))
                if self.model_type is VehicleModelType.MB or \
                    self.model_type is VehicleModelType.DS or \
                    self.model_type is VehicleModelType.DSD:
                    planner_state_c.x = planner_state_c.x - params.a * np.cos(planner_state_c.phi)
                    planner_state_c.y = planner_state_c.y - params.a * np.sin(planner_state_c.phi)
                planner_state_f = self.road.transform_trajectory_c2f(planner_state_c, initial_guess_s=ini_s)
                planner_state_f = planner_state_f.get_as_array()[:,0]
                planner_states_f.append(planner_state_f)

            first_idx = np.argmax(np.array([state[0] for state in planner_states_f]))
            if not ego_first and first_idx == 0:
                self.time2lead = copy(t_sim)
                ego_first = True

            # solve all planners for current setting
            for i_planner, [planner, container, current_state_f, current_state_c] in (
                    enumerate(zip(self.planners, self.planning_containers, planner_states_f, self.current_states_c))):
                t_start = time.time()

                # for i_state, state in enumerate(local_states):
                # c_state = self.road.transform_trajectory_f2c(FrenetTrajectory(state))
                # f_state = self.road.transform_trajectory_c2f(c_state, initial_guess_s=state[0])
                # f_state = self.road.transform_trajectory_c2f_fast(c_state)
                # local_states[i_state] = f_state.get_as_array().flatten()
                other_states = planner_states_f[:i_planner] + planner_states_f[i_planner + 1:]

                # set states and actions

                # planner.set_road(s_grid=self.road.s_grid_, kappa_grid=self.road.kappa_grid_,
                #                 nl_grid=self.road.nl_grid_, nr_grid=self.road.nr_grid_)
                planner.set_states(states_ego=planner_states_f[i_planner],
                                   states_opp=other_states,
                                   t_current=t_sim)

                # main solver call
                self.solver_wrapper_time.append(time.time() - t_start)
                planner.solve()
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
                u_current = planner.u_full[:, 0]
                self.current_states_c[i_planner] = self.simulate_step(x_real_c=current_state_c,
                                                                      u_steer_rate=u_current[1],
                                                                      u_acc=u_current[0],
                                                                      func_sim=self.funcs_sim[i_planner],
                                                                      func_sim_ls=self.funcs_sim_ls[i_planner])

                # get state
                x_full = np.zeros_like(planning_datas[-1].x)
                x_full[:, :-self.options.n_idx_skip] = planning_datas[-1].x[:, self.options.n_idx_skip:]
                x_full[:, -self.options.n_idx_skip:] = np.expand_dims(x_full[:, -self.options.n_idx_skip - 1], 1)
                full_trajectories.append(x_full)

            # we need a second loop here because all opponents move simultaneously
            # after all trajectories were computed we perform some checks
            self.full_trajectories = full_trajectories
            for i_planner in range(len(self.planners)):
                # Save timings
                if self.options.get_statistics:
                    [t_sim, t_qp, t_glob, t_reg, t_lin, qp_iter] = self.planners[i_planner].get_statistics(["time_sim",
                                                                                                            "time_qp",
                                                                                                            "time_glob",
                                                                                                            "time_reg",
                                                                                                            "time_lin",
                                                                                                            "qp_iter"])
                    self.statistics.add(computation_time=timings[i_planner],
                                        collision=None,
                                        time_sim=t_sim,
                                        time_qp=t_qp,
                                        time_glob=t_glob,
                                        time_reg=t_reg,
                                        time_lin=t_lin,
                                        qp_iter=qp_iter)
            self.max_distance_reached.append(planner_states_f[0][0])

    # u_acados = u_nlp[:, 0]
    # u = np.flip(u)  # attention, inputs are exchanged!
    # u[1] = u[1] / self.controller.veh_params.mass
    def simulate_step(self, x_real_c: np.ndarray, u_steer_rate: float, u_acc: float, func_sim, func_sim_ls):
        dt = self.td
        t = np.arange(0, dt + 1e-6, dt / 100)
        u = np.array([u_steer_rate, u_acc])
        v_thresh = 1
        if x_real_c[3] > v_thresh:
            ode_sol = odeint(func_sim, x_real_c, t, args=(u,))[-1]
            # print(ode_sol)
            # print(u)
        else:
            low_speed_state = init_ks([*x_real_c[:5], 0, 0])
            low_speed_ode_sol = \
                odeint(func_sim_ls, np.array(low_speed_state), t, args=(u,))[-1]
            ode_sol = np.zeros_like(x_real_c)
            ode_sol[:5] = low_speed_ode_sol[:5]
        next_x_real_c = deepcopy(ode_sol)
        return next_x_real_c

    def reset(self, initial_states: List[np.ndarray], road: Road):
        self.statistics = SimulationStatistics(n_vehicles=self.n_vehicles, n_sim=self.options.n_sim,
                                               delta_t=self.td * self.options.n_idx_skip)
        initial_states_c = []
        for state, p,  init_fun in zip(initial_states, self.vehicle_parameters,self.init_fun_):
            initial_states_f = FrenetTrajectory(state)
            initial_states_c.append(road.transform_trajectory_f2c(initial_states_f))
            self.current_states_c.append(init_fun([initial_states_c[-1].x,
                                                         initial_states_c[-1].y,
                                                         initial_states_c[-1].delta,
                                                         initial_states_c[-1].v,
                                                         initial_states_c[-1].phi,
                                                         0, 0]))  # p, delta0, vel0, Psi0, dotPsi0, beta0]))

        # reinitialize models in simulator due to kappa
        models = []
        for planner, vehicle_parameter in zip(self.planners, self.vehicle_parameters):
            models.append(
                FrenetModel(s_grid=road.s_grid_, p_kappa_np=road.kappa_grid_, params=vehicle_parameter,
                            p_nl_np=road.nl_grid_,
                            p_nr_np=road.nr_grid_, param_handler=None))

        self.current_states = initial_states
        self.planning_containers = []
        self.full_planner_time = []
        self.solver_wrapper_time = []
        self.all_statii = []
        self.max_distance_reached = []
        self.solver_times = None

        # set road in planners
        self.set_road(road=road)
        for i in range(self.n_vehicles):
            self.planning_containers.append(PlanningDataContainer())
        for i, (planner, initial_state) in enumerate(zip(self.planners, initial_states)):
            planner.reset_solver()
            if self.options.warm_start:
                planner.warm_start(states_ego=initial_state, actions=None)
            other_states = initial_states[:i] + initial_states[i + 1:]
            # planner.set_states(states_ego=initial_state, actions=action, states_opp=other_states)

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
