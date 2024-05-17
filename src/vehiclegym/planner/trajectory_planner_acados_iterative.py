from copy import copy, deepcopy
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from vehiclegym.utils.automotive_datastructures import FrenetTrajectory
from vehiclegym.utils.helpers import get_tracking_state
from vehiclegym.planner.planner_reference_interfaces import CostInterface
from vehiclegym.planner.trajectory_planner_acados_20221 import (
    PlannerOptions,
    VehiclePlannerAcados20221,
)
from vehicle_models.model_kinematic import KinematicModelParameters
from vehiclegym.road.road import Road


class VehiclePlannerAcados2022Iter(VehiclePlannerAcados20221):
    """
    Iterative planner, that sets the track of the model in every iteration.
    The problem right now is that the changed model leads to worse RTI convergence.
    """

    def __init__(
            self,
            ego_model_params: KinematicModelParameters,
            road: Road,
            cost_interface: CostInterface,
            planner_options: PlannerOptions,
            opp_model_params: List[KinematicModelParameters] = None,
    ):
        super().__init__(ego_model_params, road, cost_interface, planner_options, opp_model_params)
        self.current_ego_s = None
        self.s_grid_model = self.road.s_grid_

    def initialize_iter(
            self,
            states_ego: np.ndarray,
            actions: np.ndarray,
            s_grid: np.ndarray,
            kappa_grid: np.ndarray,
            nl_grid: np.ndarray,
            nr_grid: np.ndarray,
    ):
        self.current_ego_s = copy(states_ego[0])

        # get offset free
        states_ego = copy(states_ego)
        states_ego[0] -= self.current_ego_s
        # idx_closest_grid = np.argmin(np.abs(self.current_ego_s - s_grid))
        # self.s_closest_grid = s_grid[idx_closest_grid]
        # states_ego[0] -= self.s_closest_grid
        # s_grid = copy(s_grid) - self.s_closest_grid

        # casadi models are defined with a fixed s-grid
        kappa_grid_int = np.interp(
            x=self.s_grid_model + self.current_ego_s, xp=s_grid, fp=kappa_grid, left=0, right=0
        )
        nl_grid_int = np.interp(
            x=self.s_grid_model + self.current_ego_s, xp=s_grid, fp=nl_grid, left=100, right=100
        )
        nr_grid_int = np.interp(
            x=self.s_grid_model + self.current_ego_s, xp=s_grid, fp=nr_grid, left=100, right=100
        )

        # set road offset free and warm start
        self.set_road(
            s_grid=self.s_grid_model,
            kappa_grid=kappa_grid_int,
            nl_grid=nl_grid_int,
            nr_grid=nr_grid_int,
        )
        self.warm_start(states_ego=states_ego, actions=actions)

    def solve_iter(
            self,
            states_ego: np.ndarray,
            states_opp: List[np.ndarray],
            actions: np.ndarray,
            s_grid: np.ndarray,
            kappa_grid: np.ndarray,
            nl_grid: np.ndarray,
            nr_grid: np.ndarray,
    ):
        assert self.current_ego_s is not None
        self.current_ego_s = copy(states_ego[0])

        # get offset free
        states_ego = copy(states_ego)
        states_ego[0] -= self.current_ego_s
        idx_closest_grid = np.argmin(np.abs(self.current_ego_s - s_grid))
        #self.s_closest_grid = s_grid[idx_closest_grid]
        #states_ego[0] -= self.s_closest_grid
        #s_grid = copy(s_grid) - self.s_closest_grid

        states_opp = deepcopy(states_opp)
        for state_opp in states_opp:
            state_opp[0] -= self.s_closest_grid

        # casadi models are defined with a fixed s-grid
        kappa_grid_int = np.interp(
            x=self.s_grid_model + self.current_ego_s, xp=s_grid, fp=kappa_grid, left=0, right=0
        )
        nl_grid_int = np.interp(
            x=self.s_grid_model + self.current_ego_s, xp=s_grid, fp=nl_grid, left=100, right=100
        )
        nr_grid_int = np.interp(
            x=self.s_grid_model + self.current_ego_s, xp=s_grid, fp=nr_grid, left=100, right=100
        )

        # set road
        self.set_road(
            s_grid=self.s_grid_model,
            kappa_grid=kappa_grid_int,
            nl_grid=nl_grid_int,
            nr_grid=nr_grid_int,
        )

        # set obstacles
        self.set_states(states_ego=states_ego, actions=actions, states_opp=states_opp)

        # solve
        self.solve()

        # post-processing
        # self.x_full[0, :] += self.s_closest_grid
        self.x_full[0, :] += self.current_ego_s
