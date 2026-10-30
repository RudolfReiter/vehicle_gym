"""Record actual multi-agent solves and vehicle states for the scientific viewer."""

from dataclasses import asdict
import gzip
import json
from pathlib import Path
import numpy as np
from vehicle_models.model_kinematic import KinematicModelParameters
from .simulator.circuit_race import PROTOCOL, VEHICLE_COLORS


def rounded(array):
    return np.asarray(array).round(6).tolist()


def annotate_envelopes(run):
    """Measured state envelopes at recorded instants, separate from race completion."""
    parameters = KinematicModelParameters()
    wheelbase = parameters.length_front + parameters.length_rear
    run["vehicle_model"] = dict(
        wheelbase_m=wheelbase,
        mass_kg=parameters.mass,
        drag_coefficient=0.5 * parameters.drag_force_coeff * parameters.drag_area_head * 1.29,
        rolling_force_n=parameters.mass * parameters.roll_force_coeff * 9.81,
    )
    states = np.asarray(run["states"])
    envelopes = []
    for i, vehicle in enumerate(run["vehicles"]):
        speed, delta = states[:, i, 3], states[:, i, 4]
        lateral = speed**2 * np.tan(delta) / wheelbase
        limits = vehicle["limits"]
        envelopes.append(
            dict(
                speed_min=float(speed.min()),
                speed_max=float(speed.max()),
                lateral_min=float(lateral.min()),
                lateral_max=float(lateral.max()),
                speed_excess=float(max(0, speed.max() - limits["speed_max"])),
                speed_shortfall=float(max(0, limits["speed_min"] - speed.min())),
                lateral_excess=float(
                    max(
                        0,
                        lateral.max() - limits["lateral_max"],
                        limits["lateral_min"] - lateral.min(),
                    )
                ),
            )
        )
    run["metrics"]["observed_envelopes"] = envelopes
    return run


def record_race(world, seed=20000, method="MPC", controller=None, start=0.0, progress=False):
    obs = world.reset(seed, start)
    states = [rounded(world.states)]
    frames = []
    total = 0.0
    initial = world.states.copy()
    solves = []
    max_slack = 0.0
    stop = {"kind": "time_limit", "vehicles": []}
    for _ in range(world.max_steps):
        action = {} if controller is None else controller(obs, world)
        obs, reward, end, trunc, info = world.step(**action)
        total += reward
        plans = []
        for i, plan in enumerate(info["plans"]):
            if plan is None:
                plans.append(None)
                continue
            solves.append(plan.solve_ms)
            max_slack = max(max_slack, plan.max_obstacle_slack)
            plans.append(
                dict(
                    xy=rounded(plan.states[:, :2]) if plan.success else [],
                    opponents=rounded(plan.opponent_predictions[:, :, :2]),
                    modes=list(plan.opponent_modes),
                    opponent_ids=[j for j in range(world.count) if j != i],
                    success=plan.success,
                    status=plan.status,
                    residual=float(plan.max_violation),
                    solve_ms=round(plan.solve_ms, 3),
                    obstacle_slack=float(plan.max_obstacle_slack),
                )
            )
        applied = not info["failed_vehicles"]
        frames.append(
            dict(
                time=(len(states) - 1) * world.config.dt,
                plans=plans,
                references=info["references"],
                commands=rounded(info["controls"]),
                applied=applied,
            )
        )
        if applied:
            states.append(rounded(world.states))
        if end:
            stop = dict(
                kind="solver_failure"
                if info["failed_vehicles"]
                else "collision"
                if info["collision_pairs"]
                else "track_exit",
                vehicles=info["failed_vehicles"]
                or sorted(set(sum((list(pair) for pair in info["collision_pairs"]), [])))
                or info["outside_vehicles"],
            )
        if progress and world.iteration % 100 == 0:
            print(f"{method} seed={seed} t={world.iteration * world.config.dt:.1f}s", flush=True)
        if end or trunc:
            break
    vehicles = [
        dict(
            id=i,
            number=f"{i + 1:02}",
            name=method if i == 0 else f"MPC {i + 1:02}",
            controller=method if i == 0 else "MPC",
            color=VEHICLE_COLORS[i],
            limits=asdict(world.limits[i]),
            reference=None if i == 0 and method == "SAC" else asdict(world.profiles[i]),
        )
        for i in range(world.count)
    ]
    metrics = dict(
        return_value=total,
        duration_s=world.iteration * world.config.dt,
        progress_m=float(world.states[0, 0] - initial[0, 0]),
        rank=1 + int(sum(world.states[1:, 0] > world.states[0, 0])),
        completed=stop["kind"] == "time_limit",
        stop=stop,
        mean_solve_ms=float(np.mean(solves)),
        max_obstacle_slack=max_slack,
        passes=sum(e["kind"] == "pass" and e["vehicle"] == 0 for e in world.events),
        laps=float((world.states[0, 0] - initial[0, 0]) / world.track.length),
        total_passes=sum(e["kind"] == "pass" for e in world.events),
        lap_traffic_passes=sum(
            e["kind"] == "pass" and e.get("lap_traffic", False) for e in world.events
        ),
        passed_by=sum(e["kind"] == "pass" and e["opponent"] == 0 for e in world.events),
        mpc_solves=len(solves),
    )
    return annotate_envelopes(
        dict(
            protocol=PROTOCOL,
            method=method,
            seed=seed,
            dt=world.config.dt,
            horizon=world.config.dt * world.config.intervals,
            duration_limit_s=world.max_steps * world.config.dt,
            vehicles=vehicles,
            states=states,
            frames=frames,
            events=world.events,
            metrics=metrics,
            track=world.track.export_circuit(),
            planner=asdict(world.config),
        )
    )


def write_replay(run, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(run, separators=(",", ":"), allow_nan=False).encode()
    path.write_bytes(gzip.compress(payload, compresslevel=9, mtime=0))


def write_manifest(root, kind, runs, records, training=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    manifest = dict(kind=kind, protocol=PROTOCOL, runs=runs, evaluation=records, training=training)
    content = json.dumps(manifest, separators=(",", ":"), allow_nan=False)
    (root / "manifest.json").write_text(content + "\n")
    (root / "manifest.js").write_text("window.RACE_MANIFEST=" + content + ";\n")
