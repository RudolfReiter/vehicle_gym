"""Validate recorded controllers, horizon timing, finite states and evaluation rows."""

import gzip
import json
from pathlib import Path
import numpy as np
from vehicle_models.model_kinematic import KinematicModelParameters

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "docs/data/manifest.json").read_text())
assert manifest["protocol"] == "six-car-circuit-v2"
assert len(manifest["evaluation"]) == len(manifest["runs"])
for entry, row in zip(manifest["runs"], manifest["evaluation"]):
    run = json.loads(gzip.decompress((root / "docs/data" / entry["file"]).read_bytes()))
    assert run["method"] == entry["method"] == row["method"]
    assert run["seed"] == entry["seed"] == row["seed"]
    assert run["track"]["name"] == entry["track"] == row["track"]
    assert run["metrics"] == entry["metrics"] == row["metrics"]
    if entry.get("execution") and run["method"] != "MPC":
        assert run["execution"]["mode"] == entry["execution"]
    assert run["metrics"]["mpc_solves"] == (5 if run["method"] == "SAC" else 6) * len(run["frames"])
    assert len({v["limits"]["speed_max"] for v in run["vehicles"]}) > 1
    assert len({v["limits"]["longitudinal_max"] for v in run["vehicles"]}) > 1
    assert np.isclose(run["metrics"]["laps"], run["metrics"]["progress_m"] / run["track"]["length"])
    assert run["metrics"]["total_passes"] == sum(e["kind"] == "pass" for e in run["events"])
    states = np.asarray(run["states"])
    assert states.shape[1:] == (6, 5) and np.isfinite(states).all()
    parameters = KinematicModelParameters()
    commands = np.asarray(
        [frame["commands"] for frame in run["frames"] if frame["applied"]]
    ).reshape(-1, 6, 2)
    speed = states[: len(commands), :, 3]
    resistance = (
        0.5 * parameters.drag_force_coeff * parameters.drag_area_head * 1.29 * speed**2
        + parameters.mass * parameters.roll_force_coeff * 9.81
    )
    acceleration = (commands[:, :, 0] - resistance) / parameters.mass
    lower = np.array([v["limits"]["longitudinal_min"] for v in run["vehicles"]])
    upper = np.array([v["limits"]["longitudinal_max"] for v in run["vehicles"]])
    assert np.all(acceleration >= lower - 0.025), "Applied braking exceeds the recorded envelope"
    assert np.all(acceleration <= upper + 0.025), (
        "Applied acceleration exceeds the recorded envelope"
    )
    assert np.isclose(run["metrics"]["duration_s"], (len(states) - 1) * run["dt"])
    assert len(run["frames"]) in (len(states) - 1, len(states))
    assert all(v["controller"] == "MPC" for v in run["vehicles"][1:])
    assert run["vehicles"][0]["controller"] == run["method"]
    for k, frame in enumerate(run["frames"]):
        assert np.isclose(frame["time"], k * run["dt"])
        assert len(frame["plans"]) == len(frame["commands"]) == 6
        assert np.isfinite(frame["commands"]).all()
        for vehicle, plan in enumerate(frame["plans"]):
            if vehicle == 0 and run["method"] == "SAC":
                assert plan is None
                assert frame["references"][0] is None
                assert run["vehicles"][0]["reference"] is None
                continue
            assert plan is not None
            assert plan["opponent_ids"] == [i for i in range(6) if i != vehicle]
            assert np.shape(plan["opponents"]) == (5, run["planner"]["intervals"] + 1, 2)
            assert len(plan["modes"]) == 5
            if plan["success"]:
                assert np.shape(plan["xy"]) == (run["planner"]["intervals"] + 1, 2)
                np.testing.assert_allclose(plan["xy"][0], states[k, vehicle, :2], atol=2e-5)
        assert frame["applied"] == all(p is None or p["success"] for p in frame["plans"])
    assert not any(word in run for word in ["ghost", "audit", "scenario"])
    print(
        f"{entry['track']} {entry['method']} seed={entry['seed']}: {len(states) - 1} applied steps, six controller records per step"
    )
print("Recorded race data verified.")
