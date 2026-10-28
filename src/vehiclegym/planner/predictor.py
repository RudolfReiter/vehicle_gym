from typing import List
import casadi as cs
import numpy as np
from vehiclegym.road.road import Road
from vehiclegym.utils.automotive_datastructures import FrenetTrajectory


class Predictor:
    def __init__(self, road: Road, opponent_params: List, ego_model, t_grid: np.ndarray):
        self.n_opp = len(opponent_params)
        self.opp_params = opponent_params
        self.ego_model = ego_model
        self.road = road
        self.n_oversample = 1
        self.N = len(t_grid)
        # states
        s = cs.MX.sym('s')
        n = cs.MX.sym('n')
        v = cs.MX.sym('v')
        alpha_state = cs.MX.sym('alpha')
        states = cs.vertcat(s, n, v, alpha_state)

        # parameter
        Fd = cs.MX.sym('Fd')
        mass = cs.MX.sym('mass')
        kappa = cs.MX.sym('kappa', road.s_grid_.__len__())
        params = cs.vertcat(Fd, mass, kappa)

        # road interpolation
        interp_s2kappa = cs.interpolant("interp_s2kappa", "linear", [road.s_grid_])

        rhs = cs.vertcat(v * cs.cos(alpha_state) / (1 - n * interp_s2kappa(s, kappa)),
                         v * cs.sin(alpha_state),
                         1 / mass * Fd,
                         -10 * alpha_state * cs.fabs(alpha_state))

        dae = {'x': states, 'p': params, 'ode': rhs}

        opts = {}
        opts["t0"] = 0
        opts["grid"] = t_grid
        opts["output_t0"] = True
        self.integrator = cs.integrator('integrator', 'rk', dae, opts)

    def predict_linear(self, i_opponent: int, x0: FrenetTrajectory, kappa, nl, nr, acc_estimated: float = 0.):
        sol = self.integrator(x0=[x0.s, x0.n, x0.v, x0.alpha], p=[acc_estimated, 1, *kappa])

        predicted_trajectory = FrenetTrajectory()
        predicted_trajectory.s = np.array(sol['xf'])[0, ::self.n_oversample]
        predicted_trajectory.n = np.array(sol['xf'])[1, ::self.n_oversample]
        predicted_trajectory.v = np.array(sol['xf'])[2, ::self.n_oversample]
        predicted_trajectory.alpha = np.ones_like(predicted_trajectory.s) // self.n_oversample * x0.alpha
        predicted_trajectory.delta = np.zeros_like(predicted_trajectory.s) // self.n_oversample

        idx_stand_still = np.argwhere(predicted_trajectory.v < 0)

        if idx_stand_still.__len__() > 0:
            predicted_trajectory.v[idx_stand_still] = 0
            predicted_trajectory.s[idx_stand_still] = predicted_trajectory.s[idx_stand_still[0] - 1]
            predicted_trajectory.n[idx_stand_still] = predicted_trajectory.n[idx_stand_still[0] - 1]

        # clip along road boundary # check if vehicle is out of track state first
        # print("Opp:{}, smin: {}, smax: {}".format(i_opponent, predicted_trajectory.s[0], predicted_trajectory.s[-1]))
        bounds_nl = np.interp(x=predicted_trajectory.s, xp=self.road.s_grid_, fp=nl, left=20, right=20)
        bounds_nr = np.interp(x=predicted_trajectory.s, xp=self.road.s_grid_, fp=nr, left=20, right=20)
        max_bound = np.max(bounds_nl) + 0.1

        if np.abs(predicted_trajectory.n[0]) < max_bound:
            predicted_trajectory.n = np.minimum(predicted_trajectory.n,
                                                bounds_nl - self.ego_model.params_.safety_radius_boundary)
            predicted_trajectory.n = np.maximum(predicted_trajectory.n,
                                                - bounds_nr + self.ego_model.params_.safety_radius_boundary)

        return predicted_trajectory

    def predict_minimal(self, i_opponent: int, kappa,nl,nr, x0: FrenetTrajectory):
        F_min = -self.opp_params[i_opponent].maximum_deceleration_force
        mass = self.opp_params[i_opponent].mass
        sol = self.integrator(x0=[x0.s, x0.n, x0.v, x0.alpha], p=[F_min, mass, *kappa])

        predicted_trajectory = FrenetTrajectory()
        predicted_trajectory.s = np.array(sol['xf'])[0, ::self.n_oversample]
        predicted_trajectory.n = np.array(sol['xf'])[1, ::self.n_oversample]
        predicted_trajectory.v = np.array(sol['xf'])[2, ::self.n_oversample]
        predicted_trajectory.alpha = np.ones_like(predicted_trajectory.s) // self.n_oversample * x0.alpha
        predicted_trajectory.delta = np.zeros_like(predicted_trajectory.s) // self.n_oversample

        idx_stand_still = np.argwhere(predicted_trajectory.v < 0)

        if idx_stand_still.__len__() > 0:
            predicted_trajectory.v[idx_stand_still] = 0
            predicted_trajectory.s[idx_stand_still] = predicted_trajectory.s[idx_stand_still[0] - 1]
            predicted_trajectory.n[idx_stand_still] = predicted_trajectory.n[idx_stand_still[0] - 1]

        # clip along road boundary # check if vehicle is out of track state first
        bounds_nl = np.interp(x=predicted_trajectory.s, xp=self.road.s_grid_, fp=nl, left=20, right=20)
        bounds_nr = np.interp(x=predicted_trajectory.s, xp=self.road.s_grid_, fp=nr, left=20, right=20)
        max_bound = np.max(bounds_nl) + 0.1
        if np.abs(predicted_trajectory.n[0]) < max_bound:
            predicted_trajectory.n = np.minimum(predicted_trajectory.n,
                                                bounds_nl - self.ego_model.params_.safety_radius_boundary)
            predicted_trajectory.n = np.maximum(predicted_trajectory.n,
                                                - bounds_nr + self.ego_model.params_.safety_radius_boundary)

        return predicted_trajectory

    def predict_numerically(self, x0: FrenetTrajectory):

        predicted_trajectory = FrenetTrajectory()
        predicted_trajectory.s = np.ones((self.N,)) * x0.s
        predicted_trajectory.n = np.ones((self.N,)) * x0.n
        predicted_trajectory.alpha = np.ones((self.N,)) * x0.alpha
        predicted_trajectory.v = np.ones((self.N,)) * x0.v
        predicted_trajectory.delta = np.ones((self.N,)) * x0.delta

        return predicted_trajectory
