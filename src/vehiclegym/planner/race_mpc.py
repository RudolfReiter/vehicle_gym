"""Independent acados racing MPC instances with measured-state obstacle assumptions.

The original Frenet dynamics are reused. Every car has its own SQP solve and warm
start. Predicted opponent controls are never read from another car's MPC solution.
"""

from dataclasses import dataclass
import hashlib
import fcntl
from pathlib import Path
from time import perf_counter
import numpy as np
import casadi as ca
from vehicle_models.model_kinematic import KinematicModelParameters
from vehicle_models.model_kinematic_frenet import FrenetModel
from .opponent_prediction import predict_opponents
from .portable import Plan


@dataclass(frozen=True)
class RaceConfig:
    dt: float = 0.2
    intervals: int = 25
    opponents: int = 5
    speed_limit: float = 40.0
    lateral_acceleration: float = 9.0
    terminal_speed: float = 15.0
    obstacle_longitudinal: float = 9.0
    obstacle_lateral: float = 3.8
    braking: float = 10.0


class RaceMPC:
    def __init__(self, track, config=None, cache="artifacts/acados"):
        from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

        self.track, self.config = track, config or RaceConfig()
        c = self.config
        self.model = FrenetModel(
            track.s, None, KinematicModelParameters(), p_kappa_np=track.curvature
        )
        x, u = ca.MX.sym("x", 5), ca.MX.sym("u", 2)
        p = ca.MX.sym("p", 6 + 2 * c.opponents)
        # [s_ref,n_ref,v_ref,w_n,w_v,terminal_speed,opponent positions...]
        f = self.model.f_ode(x, ca.vertcat(u[0] * 1000, u[1]))
        key = hashlib.sha256(
            (track.name + repr(c) + "race-v5").encode() + track.curvature.tobytes()
        ).hexdigest()[:12]
        directory = Path(cache).resolve() / key
        directory.mkdir(parents=True, exist_ok=True)
        ocp = AcadosOcp()
        model = AcadosModel()
        model.name = "circuit_" + key
        model.x, model.u, model.p = x, u, p
        model.xdot = ca.MX.sym("xdot", 5)
        model.f_expl_expr = f
        model.f_impl_expr = model.xdot - f
        residual = ca.vertcat(
            np.sqrt(0.02) * (x[0] - p[0]),
            ca.sqrt(p[3]) * (x[1] - p[1]),
            np.sqrt(20) * x[2],
            ca.sqrt(p[4]) * (x[3] - p[2]),
            np.sqrt(2) * x[4],
        )
        model.cost_y_expr = ca.vertcat(residual, 0.10 * u[0], np.sqrt(40) * u[1])
        model.cost_y_expr_e = residual
        h = ca.vertcat(
            x[1],
            x[2],
            x[3],
            x[4],
            self.model.f_a_lat(x),
            *[
                ((x[0] - p[6 + 2 * j]) / c.obstacle_longitudinal) ** 2
                + ((x[1] - p[7 + 2 * j]) / c.obstacle_lateral) ** 2
                for j in range(c.opponents)
            ],
        )
        model.con_h_expr = h
        model.con_h_expr_e = ca.vertcat(h, x[2], x[3])
        ocp.model = model
        ocp.solver_options.N_horizon = c.intervals
        ocp.cost.cost_type = ocp.cost.cost_type_e = "NONLINEAR_LS"
        ocp.cost.W = np.eye(7)
        ocp.cost.W_e = np.eye(5)
        ocp.cost.yref = np.zeros(7)
        ocp.cost.yref_e = np.zeros(5)
        low = np.r_[
            -track.half_width + 1.1, -0.45, 0, -0.3, -c.lateral_acceleration, np.ones(c.opponents)
        ]
        high = np.r_[
            track.half_width - 1.1,
            0.45,
            c.speed_limit,
            0.3,
            c.lateral_acceleration,
            np.full(c.opponents, 1e8),
        ]
        ocp.constraints.lh = low
        ocp.constraints.uh = high
        ocp.constraints.lh_e = np.r_[low, -0.02, 0]
        ocp.constraints.uh_e = np.r_[high, 0.02, c.terminal_speed]
        ocp.constraints.idxsh = np.arange(len(low))
        ocp.constraints.idxsh_e = np.arange(len(low) + 2)
        penalties = np.r_[2e4, 2e4, 2e3, 2e4, 2e3, np.full(c.opponents, 2e5)]
        ocp.cost.zl = ocp.cost.zu = penalties
        ocp.cost.Zl = ocp.cost.Zu = penalties * 10
        ocp.cost.zl_e = ocp.cost.zu_e = np.r_[penalties, 2e4, 1e3]
        ocp.cost.Zl_e = ocp.cost.Zu_e = np.r_[penalties * 10, 2e5, 1e4]
        ocp.constraints.idxbu = np.array([0, 1])
        ocp.constraints.lbu = np.array([-12.0, -0.3])
        ocp.constraints.ubu = np.array([8.0, 0.3])
        ocp.constraints.x0 = np.array([0.0, 0, 0, 20, 0])
        ocp.parameter_values = np.r_[0, 0, 30, 0.3, 1, 18, np.tile([10000.0, 0], c.opponents)]
        ocp.solver_options.tf = c.dt * c.intervals
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.sim_method_num_stages = 4
        ocp.solver_options.sim_method_num_steps = 2
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.qp_solver_cond_N = 5
        ocp.solver_options.nlp_solver_type = "SQP"
        ocp.solver_options.nlp_solver_max_iter = 35
        ocp.solver_options.nlp_solver_tol_stat = 1.0
        ocp.solver_options.nlp_solver_tol_eq = 1e-4
        ocp.solver_options.nlp_solver_tol_ineq = 1e-4
        ocp.solver_options.nlp_solver_tol_comp = 1e-3
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.levenberg_marquardt = 1e-2
        ocp.solver_options.globalization = "MERIT_BACKTRACKING"
        ocp.solver_options.print_level = 0
        ocp.code_export_directory = str(directory / "code")
        json_file = str(directory / "ocp.json")
        # Several processes may train on the same circuit; compile exactly once.
        with (directory / "build.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            built = (directory / "code" / ("libacados_ocp_solver_" + model.name + ".so")).exists()
            self.solver = AcadosOcpSolver(
                ocp, json_file=json_file, generate=not built, build=not built
            )
        self.previous = None
        self.previous_controls = None
        dt = ca.MX.sym("dt")
        k1 = f
        k2 = self.model.f_ode(x + dt * k1 / 2, ca.vertcat(u[0] * 1000, u[1]))
        k3 = self.model.f_ode(x + dt * k2 / 2, ca.vertcat(u[0] * 1000, u[1]))
        k4 = self.model.f_ode(x + dt * k3, ca.vertcat(u[0] * 1000, u[1]))
        self.integrate = ca.Function(
            "integrate_race", [x, u, dt], [x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6]
        )

    def reset(self):
        self.previous = None
        self.previous_controls = None
        self.solver.reset()

    def plan(self, state, opponents, reference):
        c = self.config
        state = np.asarray(state)
        opponents = self.track.nearby(state, opponents)
        predictions, modes = predict_opponents(
            state, opponents, np.arange(c.intervals + 1) * c.dt, self.track, braking=c.braking
        )
        obstacles = np.tile([state[0] + 10000, 0], (c.opponents, c.intervals + 1, 1))
        obstacles[: len(opponents)] = predictions[:, :, :2]
        solver = self.solver
        started = perf_counter()
        for k in range(c.intervals + 1):
            solver.set(
                k,
                "p",
                np.r_[
                    state[0] + k * c.dt * reference.speed,
                    reference.lateral,
                    reference.speed,
                    reference.lateral_weight,
                    reference.speed_weight,
                    c.terminal_speed,
                    obstacles[:, k].ravel(),
                ],
            )
            if self.previous is None:
                guess = state.copy()
                guess[0] += k * c.dt * state[3]
                guess[2] = 0
                guess[4] = np.arctan(3 * np.interp(guess[0], self.track.s, self.track.curvature))
            else:
                guess = self.previous[min(k + 1, c.intervals)]
            solver.set(k, "x", guess)
            if k < c.intervals:
                solver.set(
                    k,
                    "u",
                    np.array([0.6, 0.0])
                    if self.previous_controls is None
                    else self.previous_controls[min(k + 1, c.intervals - 1)],
                )
        solver.set(0, "lbx", state)
        solver.set(0, "ubx", state)
        candidates = []
        for attempt in range(4):
            if attempt:
                # Different initial guesses for the same OCP, never alternate controls.
                solver.reset()
                lateral = [state[1], -track_margin(self.track), track_margin(self.track)][
                    attempt - 1
                ]
                for k in range(c.intervals + 1):
                    t = k * c.dt
                    blend = 1 - np.exp(-t * t / 2)
                    guess = state.copy()
                    guess[0] += max(0, state[3] * t - 1.5 * t * t)
                    guess[1] = state[1] + blend * (lateral - state[1])
                    guess[2] = np.arctan2(
                        (lateral - state[1]) * t * np.exp(-t * t / 2), max(5, state[3] - 3 * t)
                    )
                    guess[3] = max(5, state[3] - 3 * t)
                    guess[4] = np.arctan(
                        3 * np.interp(guess[0], self.track.s, self.track.curvature)
                    )
                    solver.set(k, "x", guess)
                    if k < c.intervals:
                        solver.set(k, "u", np.array([-3.5, 0.0]))
                solver.set(0, "lbx", state)
                solver.set(0, "ubx", state)
            status = solver.solve()
            states = np.array([solver.get(k, "x") for k in range(c.intervals + 1)])
            controls = np.array([solver.get(k, "u") for k in range(c.intervals)])
            residuals = np.asarray(solver.get_residuals())
            slack = max(
                float(
                    np.max(
                        solver.get(k, "sl")[-c.opponents :]
                        if k < c.intervals
                        else solver.get(k, "sl")[5 : 5 + c.opponents]
                    )
                )
                for k in range(1, c.intervals + 1)
            )
            violation = float(np.max(residuals[1:3]))
            success = bool(
                status in (0, 2)
                and np.isfinite(states).all()
                and np.isfinite(controls).all()
                and violation < 0.02
            )
            candidates.append(
                (
                    not success,
                    max(slack, 0) + violation,
                    solver.get_cost(),
                    states,
                    controls,
                    status,
                    residuals,
                    slack,
                    success,
                )
            )
            if success and slack < 0.01:
                break
        _, _, _, states, controls, status, residuals, slack, success = min(
            candidates, key=lambda item: item[:3]
        )
        self.previous = states.copy() if success else None
        self.previous_controls = controls.copy() if success else None
        controls[:, 0] *= 1000
        return Plan(
            states,
            controls,
            success,
            f"acados status {status}",
            float(np.max(residuals[1:3])),
            (perf_counter() - started) * 1000,
            slack,
            predictions,
            modes,
        )


def track_margin(track):
    return track.half_width - 2.2
