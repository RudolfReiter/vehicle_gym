"""Seeded interactive racing with a receding-horizon ego MPC.

Opponents follow a fixed, gap-seeking passing policy with lane commitment and
car following. They are simulated vehicles, not replayed obstacle trajectories.
The ego sees their measured states, never their future controls or intent.
"""

from dataclasses import asdict
import numpy as np
from vehiclegym.road.track import Track
from vehiclegym.planner.portable import PredictivePlanner, PlannerConfig
from vehiclegym.utils.evaluation_tools import check_collision_cartesian

SCENARIOS = ("overtaking", "blocking", "mixed")
ENVIRONMENT_VERSION = "interactive-v2"


class RacingWorld:
    def __init__(self, config=None, max_steps=96):
        self.config = config or PlannerConfig()
        self.max_steps = max_steps
        self.planner = None
        self.finished = True

    def reset(self, seed=0, scenario="overtaking", bend=None):
        if scenario not in SCENARIOS:
            raise ValueError(f"Scenario must be one of {SCENARIOS}")
        rng = np.random.default_rng(seed)
        self.track = Track.synthetic(rng.uniform(-0.0025, 0.0025) if bend is None else bend)
        self.planner = PredictivePlanner(self.track, self.config)
        self.scenario, self.seed, self.iteration = scenario, seed, 0
        side = rng.choice([-1.0, 1.0])
        self.state = np.array([0.0, rng.uniform(-0.3, 0.3), 0.0, rng.uniform(29, 31), 0.0])
        if scenario == "overtaking":
            distances, lateral, speeds = [23, 49, 76], [0, 3.5, -3.5], [25, 26, 27]
        elif scenario == "blocking":
            distances, lateral, speeds = [-20, -43, -66], [3.3, -3.3, 0], [36, 35, 34]
        else:
            distances, lateral, speeds = [26, -22, 59], [0, 3.3, -3.5], [25, 36, 27]
        self.opponents = np.zeros((3, 5))
        self.opponents[:, 0] = np.array(distances) + rng.uniform(-2, 2, 3)
        self.opponents[:, 1] = side * np.array(lateral) + rng.uniform(-0.15, 0.15, 3)
        self.speed_targets = np.array(speeds) + rng.uniform(-0.4, 0.4, 3)
        self.opponents[:, 3] = np.minimum(self.speed_targets, 34.0)
        self.lane_targets = self.opponents[:, 1].copy()
        self.lane_commitment = np.zeros(3)
        self.preferred_sides = side * np.array([1, -1, 1])
        self.order = np.sign(self.state[0] - self.opponents[:, 0])
        self.events = []
        self.ego_history = [self.state.copy()]
        self.opponent_history = [self.opponents.copy()]
        self.last_defense = -10.0
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
        controls, held_up, attacking = [], [], []
        all_states = np.vstack((self.state, self.opponents))
        now = self.iteration * self.config.dt
        for i, state in enumerate(self.opponents):
            others = np.delete(all_states, i + 1, axis=0)
            gaps = others[:, 0] - state[0]
            slower_ahead = (gaps > 0) & (gaps < 48) & (others[:, 3] < self.speed_targets[i] - 0.5)
            if np.any(slower_ahead) and now >= self.lane_commitment[i]:
                candidates = np.array([-4.2, 0.0, 4.2])
                scores = []
                for lane in candidates:
                    possible_speed = self.speed_targets[i]
                    for gap, other in zip(gaps, others):
                        if 0 < gap < 48 and abs(other[1] - lane) < 2.8:
                            possible_speed = min(possible_speed, other[3] + 0.7 * (gap - 10))
                        crossing = min(state[1], lane) - 2 < other[1] < max(state[1], lane) + 2
                        if abs(gap) < 9 and crossing:
                            possible_speed -= 40
                    score = (
                        possible_speed
                        - 0.12 * abs(lane - state[1])
                        + 0.03 * lane * self.preferred_sides[i]
                    )
                    scores.append(score)
                target = candidates[int(np.argmax(scores))]
                if abs(target - self.lane_targets[i]) > 1:
                    self.lane_targets[i] = target
                    self.lane_commitment[i] = now + 2.5
            speed = self.speed_targets[i]
            for gap, other in zip(gaps, others):
                projected_n = other[1] + 0.6 * other[3] * np.sin(other[2])
                aligned = min(abs(other[1] - state[1]), abs(projected_n - state[1])) < 2.7
                if 0 < gap < 48 and aligned:
                    speed = min(speed, max(0.0, other[3] + 0.85 * (gap - 10)))
            acceleration = np.clip(2 * (speed - state[3]), -10, 5)
            desired_alpha = np.arctan2(1.4 * (self.lane_targets[i] - state[1]), max(state[3], 2))
            curvature = np.interp(state[0], self.track.s, self.track.curvature)
            desired_delta = np.arctan(
                3 * (curvature + 2 * (desired_alpha - state[2]) / max(state[3], 2))
            )
            lateral_limit = np.arctan(10 * 3 / max(state[3] ** 2, 4))
            desired_delta = np.clip(desired_delta, -lateral_limit, lateral_limit)
            rate = np.clip(3 * (desired_delta - state[4]), -0.39, 0.39)
            resistance = 0.5 * 0.29 * 2.07 * 1.29 * state[3] ** 2 + 1160 * 0.015 * 9.81
            controls.append([(1160 * acceleration + resistance) / 1000, rate])
            gap_to_ego = self.state[0] - state[0]
            held_up.append(
                4.6 < gap_to_ego < 30
                and abs(state[1] - self.state[1]) < 2.7
                and speed < self.speed_targets[i] - 1
                and state[3] < self.speed_targets[i] - 0.5
            )
            attacking.append(0 < gap_to_ego < 48 and self.speed_targets[i] > self.state[3] + 0.5)
        return controls, np.array(held_up, dtype=bool), np.array(attacking, dtype=bool)

    def _collision(self):
        def geometry(state):
            return self.track.position(state[0], state[1]), (
                4.6,
                1.9,
                np.interp(state[0], self.track.s, self.track.heading) + state[2],
            )

        position, dimensions = geometry(self.state)
        return any(
            check_collision_cartesian(position, dimensions, *geometry(other))
            for other in self.opponents
        )

    def step(self, reference):
        if self.finished:
            raise RuntimeError("Call reset before stepping a finished episode")
        before = self.state.copy()
        plan = self.planner.plan(self.state, self.opponents, reference)
        info = {
            "plan": plan,
            "reference": asdict(reference),
            "solver_failed": not plan.success,
            "collision": False,
            "min_body_separation_m": None,
            "held_up": [False] * 3,
            "attacking": [False] * 3,
            "defensive_move": False,
        }
        if not plan.success:
            self.finished = True
            return self.observation(), -10.0, True, False, info
        controls, held_up, attacking = self._opponent_controls()
        ego_control = plan.controls[0] / np.array([1000, 1])
        separation, collision = np.inf, False
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
            collision |= self._collision()
        self.iteration += 1
        now = self.iteration * self.config.dt
        self.ego_history.append(self.state.copy())
        self.opponent_history.append(self.opponents.copy())
        for i, other in enumerate(self.opponents):
            difference = self.state[0] - other[0]
            if abs(difference) > 4.6 and np.sign(difference) != self.order[i]:
                kind = "ego_pass" if difference > 0 else "opponent_pass"
                self.events.append({"time": now, "kind": kind, "opponent": i})
                self.order[i] = np.sign(difference)
        # Count a defensive move only when the ego actually moved laterally into
        # a challenger’s line and the challenger is measurably held below target.
        past = max(0, len(self.ego_history) - 9)
        old_ego, old_opponents = self.ego_history[past], self.opponent_history[past]
        lateral_motion = abs(self.state[1] - old_ego[1]) > 0.8
        covered_line = (
            np.abs(old_opponents[:, 1] - old_ego[1]) - np.abs(self.opponents[:, 1] - self.state[1])
            > 0.8
        )
        defense = bool(
            lateral_motion and np.any(held_up & covered_line) and now - self.last_defense > 2.5
        )
        if defense:
            i = int(np.flatnonzero(held_up & covered_line)[0])
            self.events.append({"time": now, "kind": "defensive_move", "opponent": i})
            self.last_defense = now
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
            held_up=held_up.tolist(),
            attacking=attacking.tolist(),
            defensive_move=defense,
        )
        return self.observation(), float(reward), terminated, truncated, info
