"""Seeded racing scenarios for demonstrations and high-level policy learning.

The ego uses nonlinear predictive planning. Opponents use fixed lane-tracking
and car-following controllers. These are deliberately distinct from the paper's
MPC-controlled opponents. Failed plans terminate before executing a control.
"""

from dataclasses import asdict
import numpy as np
from vehiclegym.road.track import Track
from vehiclegym.planner.portable import PredictivePlanner, PlannerConfig

SCENARIOS = ("overtaking", "blocking", "mixed")


class RacingWorld:
    def __init__(self, config=None, max_steps=80):
        self.config = config or PlannerConfig()
        self.max_steps = max_steps
        self.planner = None
        self.finished = True

    def reset(self, seed=0, scenario="overtaking", bend=None):
        if scenario not in SCENARIOS:
            raise ValueError(f"Scenario must be one of {SCENARIOS}")
        rng = np.random.default_rng(seed)
        self.track = Track.synthetic(rng.uniform(-0.0035, 0.0035) if bend is None else bend)
        self.planner = PredictivePlanner(self.track, self.config)
        self.scenario, self.seed, self.iteration = scenario, seed, 0
        self.state = np.array([0.0, rng.uniform(-0.4, 0.4), 0.0, rng.uniform(22, 25), 0.0])
        if scenario == "overtaking":
            distances, lateral, speeds = [18, 42, 65], [-1.5, 1.8, -0.5], [19, 20, 22]
        elif scenario == "blocking":
            distances, lateral, speeds = [-16, -38, -60], [2.8, -2.8, 0], [31, 30, 29]
        else:
            distances, lateral, speeds = [24, -20, 54], [-1.7, 2.7, 0.3], [20, 31, 23]
        self.opponents = np.zeros((3, 5))
        self.opponents[:, 0] = np.array(distances) + rng.uniform(-3, 3, 3)
        self.opponents[:, 1] = np.array(lateral) + rng.uniform(-0.2, 0.2, 3)
        self.opponents[:, 3] = np.array(speeds) + rng.uniform(-1, 1, 3)
        self.lane_targets = self.opponents[:, 1].copy()
        self.speed_targets = self.opponents[:, 3].copy()
        self.finished = False
        return self.observation()

    def observation(self):
        curvature = (
            np.interp(
                self.state[0] + np.array([0, 10, 20, 40, 60, 90, 120]),
                self.track.s,
                self.track.curvature,
            )
            / 0.04
        )
        ego = self.state[[1, 3, 2]] / np.array([7, 32, 0.6])
        opp = self.opponents[:, [0, 1, 3, 2]].copy()
        opp[:, 0] -= self.state[0]
        opp /= [80, 7, 32, 0.6]
        return np.r_[curvature, ego, opp.ravel()].astype(np.float32)

    def _opponent_controls(self):
        controls = []
        all_states = np.vstack((self.state, self.opponents))
        for i, state in enumerate(self.opponents):
            speed = self.speed_targets[i]
            for j, other in enumerate(all_states):
                gap = other[0] - state[0]
                if j != i + 1 and 0 < gap < 45 and abs(other[1] - state[1]) < 2.6:
                    speed = min(speed, max(0, other[3] + 0.8 * (gap - 10)))
            acceleration = np.clip(1.7 * (speed - state[3]), -9, 4)
            desired_alpha = np.arctan2(1.2 * (self.lane_targets[i] - state[1]), max(state[3], 2))
            curvature = np.interp(state[0], self.track.s, self.track.curvature)
            desired_delta = np.arctan(
                3 * (curvature + 1.8 * (desired_alpha - state[2]) / max(state[3], 2))
            )
            rate = np.clip(3 * (desired_delta - state[4]), -0.39, 0.39)
            resistance = 0.5 * 0.29 * 2.07 * 1.29 * state[3] ** 2 + 1160 * 0.015 * 9.81
            controls.append([(1160 * acceleration + resistance) / 1000, rate])
        return controls

    def step(self, reference):
        if self.finished:
            raise RuntimeError("Call reset before stepping a finished episode")
        before = self.state.copy()
        plan = self.planner.plan(self.state, self.opponents[:, [0, 1, 3]], reference)
        info = {
            "plan": plan,
            "reference": asdict(reference),
            "solver_failed": not plan.success,
            "collision": False,
            "min_body_separation_m": None,
        }
        if not plan.success:
            self.finished = True
            return self.observation(), -10.0, True, False, info
        controls = self._opponent_controls()
        ego_control = plan.controls[0] / np.array([1000, 1])
        separation = np.inf
        for _ in range(5):
            self.state = np.asarray(
                self.planner.integrate(self.state, ego_control, self.config.dt / 5)
            ).ravel()
            for i, control in enumerate(controls):
                self.opponents[i] = np.asarray(
                    self.planner.integrate(self.opponents[i], control, self.config.dt / 5)
                ).ravel()
            ds = abs(self.opponents[:, 0] - self.state[0]) - 4.6
            dn = abs(self.opponents[:, 1] - self.state[1]) - 1.9
            separation = min(separation, float(np.min(np.maximum(ds, dn))))
        self.iteration += 1
        collision = separation < 0
        outside = abs(self.state[1]) > self.track.half_width - 1.0
        reward = (self.state[0] - before[0]) / self.config.dt / 200 + int(
            sum(self.state[0] > self.opponents[:, 0])
        )
        terminated = collision or outside or not np.isfinite(self.state).all()
        truncated = self.iteration >= self.max_steps
        if terminated:
            reward -= 10
        self.finished = terminated or truncated
        info.update(
            collision=collision,
            min_body_separation_m=separation,
            outside_track=outside,
            rank=1 + int(sum(self.state[0] < self.opponents[:, 0])),
        )
        return self.observation(), float(reward), terminated, truncated, info
