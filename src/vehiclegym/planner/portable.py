"""CasADi/IPOPT research backend using the original Frenet vehicle model.

This backend checks numerical feasibility but does not claim recursive safety.
The original acados planner remains available as a separate optional backend.
"""

from dataclasses import dataclass
from time import perf_counter
import casadi as ca
import numpy as np
from vehicle_models.model_kinematic import KinematicModelParameters
from vehicle_models.model_kinematic_frenet import FrenetModel
from vehiclegym.planner.opponent_prediction import normalize_opponents, predict_opponents


@dataclass(frozen=True)
class PlannerConfig:
    intervals: int = 12
    dt: float = 0.25
    max_opponents: int = 3
    speed_limit: float = 32.0
    lateral_acceleration: float = 8.0
    obstacle_longitudinal: float = 7.0
    obstacle_lateral: float = 3.0


@dataclass
class Plan:
    states: np.ndarray
    controls: np.ndarray
    success: bool
    status: str
    max_violation: float
    solve_ms: float
    max_obstacle_slack: float
    opponent_predictions: np.ndarray
    opponent_modes: tuple


class PredictivePlanner:
    def __init__(self, track, config=None):
        self.track = track
        self.config = c = config or PlannerConfig()
        if c.intervals < 2 or c.dt <= 0 or c.max_opponents < 0:
            raise ValueError("Invalid planner horizon or opponent count")
        self.parameters = KinematicModelParameters()
        self.model = FrenetModel(track.s, None, self.parameters, p_kappa_np=track.curvature)
        x, u = ca.MX.sym("x", 5), ca.MX.sym("u", 2)
        f = self.model.f_ode
        physical_u = ca.vertcat(u[0] * 1000, u[1])  # optimizer force in kN
        dt = ca.MX.sym("dt")
        k1 = f(x, physical_u)
        k2 = f(x + dt * k1 / 2, physical_u)
        k3 = f(x + dt * k2 / 2, physical_u)
        k4 = f(x + dt * k3, physical_u)
        self.integrate = ca.Function(
            "integrate", [x, u, dt], [x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6]
        )
        self.step = ca.Function("rk4", [x, u], [self.integrate(x, u, c.dt)])
        X = ca.MX.sym("X", 5, c.intervals + 1)
        U = ca.MX.sym("U", 2, c.intervals)
        slack = ca.MX.sym("obstacle_slack", c.intervals + 1)
        P = ca.MX.sym("P", 9 + 2 * c.max_opponents * (c.intervals + 1))
        terms, low, high = [], [], []

        def constrain(expression, lower, upper):
            terms.append(ca.vec(expression))
            low.extend(np.broadcast_to(lower, expression.numel()))
            high.extend(np.broadcast_to(upper, expression.numel()))

        constrain(X[:, 0] - P[:5], 0, 0)
        cost = 0
        for k in range(c.intervals + 1):
            state = X[:, k]
            ref_s = P[0] + k * c.dt * P[5]
            cost += c.dt * (
                0.015 * (state[0] - ref_s) ** 2
                + P[8] * (state[1] - P[6]) ** 2
                + 2 * state[2] ** 2
                + P[7] * (state[3] - P[5]) ** 2
                + 2 * state[4] ** 2
            )
            constrain(state[1], -track.half_width + 1.1, track.half_width - 1.1)
            constrain(state[2], -0.6, 0.6)
            constrain(state[3], 0, c.speed_limit)
            constrain(state[4], -0.30, 0.30)
            constrain(self.model.f_a_lat(state), -c.lateral_acceleration, c.lateral_acceleration)
            for j in range(c.max_opponents):
                offset = 9 + 2 * (j * (c.intervals + 1) + k)
                opponent = P[offset : offset + 2]
                ds = (state[0] - opponent[0]) / c.obstacle_longitudinal
                dn = (state[1] - opponent[1]) / c.obstacle_lateral
                constrain(ds**2 + dn**2 + slack[k], 1, np.inf)
            cost += 1e4 * slack[k] + 1e4 * slack[k] ** 2
            if k < c.intervals:
                constrain(X[:, k + 1] - self.step(state, U[:, k]), 0, 0)
                cost += c.dt * (0.01 * U[0, k] ** 2 + 0.4 * U[1, k] ** 2)
        variables = ca.vertcat(ca.vec(X), ca.vec(U), slack)
        constraints = ca.vertcat(*terms)
        self.solver = ca.nlpsol(
            "mpp",
            "ipopt",
            {"x": variables, "p": P, "f": cost, "g": constraints},
            {
                "print_time": False,
                "ipopt.print_level": 0,
                "ipopt.sb": "yes",
                "ipopt.max_iter": 80,
                "ipopt.tol": 1e-5,
                "ipopt.acceptable_tol": 1e-4,
            },
        )
        self.constraint_function = ca.Function("constraints", [variables, P], [constraints])
        self.lower_g, self.upper_g = np.array(low), np.array(high)
        self.lower_x = np.r_[
            np.full(5 * (c.intervals + 1), -np.inf),
            np.tile([-12.0, -0.39], c.intervals),
            np.zeros(c.intervals + 1),
        ]
        self.upper_x = np.r_[
            np.full(5 * (c.intervals + 1), np.inf),
            np.tile([8.0, 0.39], c.intervals),
            np.full(c.intervals + 1, 0.08),
        ]
        self.previous = None

    def reset(self):
        self.previous = None

    def plan(self, state, opponents, reference):
        c = self.config
        state = np.asarray(state, dtype=float)
        opponents = normalize_opponents(opponents)
        if state.shape != (5,) or len(opponents) > c.max_opponents:
            raise ValueError("Expected five ego states and at most max_opponents vehicles")
        if not np.isfinite(state).all() or not np.isfinite(opponents).all():
            raise ValueError("States must be finite")
        predictions, modes = predict_opponents(
            state, opponents, np.arange(c.intervals + 1) * c.dt, self.track
        )
        obstacles = np.tile([state[0] + 10000, 0], (c.max_opponents, c.intervals + 1, 1))
        obstacles[: len(opponents)] = predictions[:, :, :2]
        p = np.r_[
            state,
            reference.speed,
            reference.lateral,
            reference.speed_weight,
            reference.lateral_weight,
            obstacles.ravel(),
        ]
        if self.previous is None:
            guess = np.tile(state, (c.intervals + 1, 1))
            guess[:, 0] += np.arange(c.intervals + 1) * c.dt * state[3]
            initial = np.r_[guess.ravel(), np.zeros(3 * c.intervals + 1)]
        else:
            initial = self.previous.copy()
        start = perf_counter()
        result = self.solver(
            x0=initial, p=p, lbx=self.lower_x, ubx=self.upper_x, lbg=self.lower_g, ubg=self.upper_g
        )
        z = np.asarray(result["x"]).ravel()
        g = np.asarray(self.constraint_function(z, p)).ravel()
        violation = max(
            0.0,
            float(np.max(self.lower_g - g)),
            float(np.max(g - self.upper_g)),
            float(np.max(self.lower_x - z)),
            float(np.max(z - self.upper_x)),
        )
        success = bool(self.solver.stats()["success"] and np.isfinite(z).all() and violation < 2e-4)
        states = z[: 5 * (c.intervals + 1)].reshape(-1, 5)
        inputs = z[5 * (c.intervals + 1) : 5 * (c.intervals + 1) + 2 * c.intervals].reshape(-1, 2)
        slack_values = z[-(c.intervals + 1) :]
        if success:
            self.previous = np.r_[
                np.vstack((states[1:], states[-1])).ravel(),
                np.vstack((inputs[1:], inputs[-1])).ravel(),
                slack_values,
            ]
        else:
            self.previous = None
        controls = inputs.copy()
        controls[:, 0] *= 1000
        return Plan(
            states,
            controls,
            success,
            self.solver.stats()["return_status"],
            violation,
            (perf_counter() - start) * 1000,
            float(np.max(slack_values)),
            predictions,
            modes,
        )
