"""Synchronous six-car racing. Every opponent is an independent racing MPC."""

from dataclasses import asdict
from itertools import combinations
import numpy as np
from vehiclegym.road.circuit import Circuit
from vehiclegym.planner.race_mpc import RaceMPC, RaceConfig
from vehiclegym.planner.references import FrenetReference
from vehiclegym.utils.evaluation_tools import check_collision_cartesian

PROTOCOL = "six-car-circuit-v1"
VEHICLE_COLORS = ["#1673c9", "#e46d36", "#48a07f", "#9a76c5", "#d5ae43", "#bb526a"]


class CircuitRace:
    def __init__(self, track="IMS", duration=180.0, config=None, planner_factory=RaceMPC):
        self.track = Circuit.load(track) if isinstance(track, str) else track
        self.config = config or RaceConfig()
        self.count = self.config.opponents + 1
        self.max_steps = round(duration / self.config.dt)
        self.planners = [planner_factory(self.track, self.config) for _ in range(self.count)]
        self.profiles = [
            FrenetReference(speed=v, lateral=n, speed_weight=1.0, lateral_weight=2.0)
            for v, n in [(28, 0), (24, -1.4), (26, 1.4), (27, 0), (31, -1), (29, 1)]
        ]
        if self.count != len(self.profiles):
            raise ValueError("This experiment protocol uses exactly six cars")
        self.finished = True

    def reset(self, seed=0, start=0.0):
        rng = np.random.default_rng(seed)
        self.seed, self.iteration = seed, 0
        offsets = np.array([0, 20, 40, 60, -20, -40], float) + rng.uniform(-2, 2, self.count)
        self.states = np.zeros((self.count, 5))
        self.states[:, 0] = start + offsets
        self.states[:, 1] = np.array([0, -1.4, 1.4, 0, -1, 1]) + rng.uniform(
            -0.15, 0.15, self.count
        )
        ahead = np.linspace(start - 50, start + 150, 100)
        max_kappa = max(
            0.001, float(np.max(abs(np.interp(ahead, self.track.s, self.track.curvature))))
        )
        self.states[:, 3] = min(23.0, np.sqrt(7.0 / max_kappa))
        kappa = np.interp(self.states[:, 0], self.track.s, self.track.curvature)
        self.states[:, 4] = np.arctan(3 * kappa / (1 - self.states[:, 1] * kappa))
        for planner in self.planners:
            planner.reset()
        self.finished = False
        self.events = []
        self.previous_order = np.sign(self.states[:, 0, None] - self.states[None, :, 0])
        self.previous_modes = {}
        self.control_source = []
        return self.observation()

    def observation(self, vehicle=0):
        state = self.states[vehicle]
        others = self.track.nearby(state, np.delete(self.states, vehicle, axis=0))
        curvature = (
            np.interp(
                state[0] + np.array([0, 10, 25, 50, 80, 120, 180]),
                self.track.s,
                self.track.curvature,
            )
            / 0.05
        )
        ego = state[[1, 3, 2, 4]] / np.array([self.track.half_width, 40, 0.45, 0.3])
        opponents = others[:, [0, 1, 3, 2]].copy()
        opponents[:, 0] -= state[0]
        opponents /= np.array([120, self.track.half_width, 40, 0.45])
        return np.r_[curvature, ego, opponents.ravel()].astype(np.float32)

    def step(self, reference=None, direct_control=None):
        if self.finished:
            raise RuntimeError("Reset before stepping a finished race")
        if reference is not None and direct_control is not None:
            raise ValueError("Choose either HILEPP references or direct SAC controls")
        snapshot = self.states.copy()
        plans = []
        references = []
        controls = []
        sources = []
        for vehicle, planner in enumerate(self.planners):
            ref = reference if vehicle == 0 and reference is not None else self.profiles[vehicle]
            references.append(asdict(ref))
            if vehicle == 0 and direct_control is not None:
                control = np.asarray(direct_control, dtype=float)
                if control.shape != (2,) or not np.isfinite(control).all():
                    raise ValueError("Direct controls must be finite force and steering rate")
                control = np.clip(control, [-12000, -0.3], [8000, 0.3])
                plans.append(None)
                controls.append(control)
                sources.append("SAC")
                continue
            plan = planner.plan(snapshot[vehicle], np.delete(snapshot, vehicle, axis=0), ref)
            plans.append(plan)
            controls.append(plan.controls[0])
            sources.append("HILEPP" if vehicle == 0 and reference is not None else "MPC")
        failures = [i for i, p in enumerate(plans) if p is not None and not p.success]
        info = dict(
            plans=plans,
            references=references,
            controls=np.array(controls),
            control_sources=sources,
            failed_vehicles=failures,
            collision_pairs=[],
            outside_vehicles=[],
            events=[],
        )
        if failures:
            self.finished = True
            return self.observation(), -10.0, True, False, info
        collisions = set()
        outside = set()
        for _ in range(4):
            for i, (planner, control) in enumerate(zip(self.planners, controls)):
                self.states[i] = np.asarray(
                    planner.integrate(self.states[i], control / [1000, 1], self.config.dt / 4)
                ).ravel()
            positions = self.track.position(self.states[:, 0], self.states[:, 1])
            headings = (
                np.interp(self.states[:, 0], self.track.s, self.track.heading) + self.states[:, 2]
            )
            for i, j in combinations(range(self.count), 2):
                if np.linalg.norm(positions[i] - positions[j]) < 8 and check_collision_cartesian(
                    positions[i], (4.6, 1.9, headings[i]), positions[j], (4.6, 1.9, headings[j])
                ):
                    collisions.add((i, j))
            for i, x in enumerate(self.states):
                if (
                    abs(x[1]) + 2.3 * abs(np.sin(x[2])) + 0.95 * abs(np.cos(x[2]))
                    > self.track.half_width
                    or x[3] < -0.05
                    or not np.isfinite(x).all()
                ):
                    outside.add(i)
        self.iteration += 1
        now = self.iteration * self.config.dt
        events = []
        for i, j in combinations(range(self.count), 2):
            difference = self.states[i, 0] - self.states[j, 0]
            if abs(difference) > 4.6 and np.sign(difference) != self.previous_order[i, j]:
                events.append(
                    dict(
                        time=now,
                        kind="pass",
                        vehicle=i if difference > 0 else j,
                        opponent=j if difference > 0 else i,
                    )
                )
                self.previous_order[i, j] = np.sign(difference)
        for i, plan in enumerate(plans):
            if plan is None:
                continue
            opponents = [j for j in range(self.count) if j != i]
            for j, mode in zip(opponents, plan.opponent_modes):
                key = (i, j)
                if key in self.previous_modes and self.previous_modes[key] != mode:
                    events.append(
                        dict(
                            time=now - self.config.dt,
                            kind="prediction_change",
                            vehicle=i,
                            opponent=j,
                            mode=mode,
                        )
                    )
                self.previous_modes[key] = mode
        self.events.extend(events)
        reward = (self.states[0, 0] - snapshot[0, 0]) / self.config.dt / 200 + sum(
            self.states[0, 0] > self.states[1:, 0]
        )
        terminated = bool(collisions or outside)
        truncated = self.iteration >= self.max_steps
        if terminated:
            reward -= 10
        self.finished = terminated or truncated
        info.update(
            collision_pairs=sorted(collisions), outside_vehicles=sorted(outside), events=events
        )
        return self.observation(), float(reward), terminated, truncated, info
