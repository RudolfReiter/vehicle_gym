"""Generate replay data from the actual model, optimizer, and simulation."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from vehiclegym.planner.references import FrenetReference
from vehiclegym.simulator.racing_world import RacingWorld, SCENARIOS


def rollout(world, controller, seed, scenario, name, identifier):
    world.reset(seed, scenario)
    states, opponents, plans, references, times = (
        [world.state.tolist()],
        [world.opponents.tolist()],
        [],
        [],
        [0.0],
    )
    total, clearance, failures, collision = 0.0, np.inf, 0, False
    for _ in range(world.max_steps):
        reference = controller(world.observation())
        _, reward, end, trunc, info = world.step(reference)
        plan = info["plan"]
        plans.append(
            {
                "origin": times[-1],
                "states": plan.states.tolist() if plan.success else [],
                "success": plan.success,
                "max_violation": plan.max_violation,
                "solve_ms": plan.solve_ms,
                "status": plan.status,
                "obstacle_slack": plan.max_obstacle_slack,
            }
        )
        references.append(asdict(reference))
        total += reward
        failures += int(info["solver_failed"])
        collision |= info["collision"]
        if not info["solver_failed"]:
            states.append(world.state.tolist())
            opponents.append(world.opponents.tolist())
            times.append(world.iteration * world.config.dt)
            clearance = min(clearance, info["min_body_separation_m"])
        if end or trunc:
            break
    metrics = {
        "return": total,
        "progress_m": float(world.state[0]),
        "final_rank": 1 + int(sum(world.opponents[:, 0] > world.state[0])),
        "duration_s": times[-1],
        "solver_failures": failures,
        "collision": bool(collision),
        "min_body_separation_m": float(clearance) if np.isfinite(clearance) else None,
        "mean_solve_ms": float(np.mean([p["solve_ms"] for p in plans])),
        "max_violation": float(max(p["max_violation"] for p in plans)),
        "max_obstacle_slack": float(max(p["obstacle_slack"] for p in plans)),
        "completed": world.iteration >= world.max_steps and not collision and not failures,
    }
    return {
        "id": identifier,
        "name": name,
        "seed": seed,
        "track": world.track.export(-100, 700, 801),
        "time": times,
        "states": states,
        "opponents": opponents,
        "plans": plans,
        "references": references,
        "metrics": metrics,
    }


def export(data, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    content = json.dumps(data, separators=(",", ":"), allow_nan=False)
    (output / "experiment.json").write_text(content + "\n")
    (output / "experiment.js").write_text("window.EXPERIMENT=" + content + ";\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("docs/data"))
    args = parser.parse_args()
    data = {"kind": "vehicle_gym", "dt": 0.25, "horizon": 3.0, "scenarios": []}
    for scenario in SCENARIOS:
        runs = []
        for identifier, name, lateral in [
            ("center", "Center reference", 0.0),
            ("left", "Left reference", 3.0),
            ("right", "Right reference", -3.0),
        ]:
            run = rollout(
                RacingWorld(max_steps=64),
                lambda obs: FrenetReference(lateral=lateral),
                101,
                scenario,
                name,
                identifier,
            )
            runs.append(run)
            print(scenario, identifier, run["metrics"], flush=True)
        data["scenarios"].append({"id": scenario, "runs": runs})
    export(data, args.output)


if __name__ == "__main__":
    main()
