"""Independent high-accuracy ODE audit of recorded commands versus measured motion.

This uses SciPy's adaptive integrator, not the simulator's fixed-step RK4 routine.
acados is not required. Ten deterministic sampled intervals per race are checked.
"""

import gzip
import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from vehicle_models.model_kinematic import KinematicModelParameters
from vehicle_models.model_kinematic_frenet import FrenetModel
from vehiclegym.road.circuit import Circuit

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "docs/data/manifest.json").read_text())
models = {}
worst = 0.0
for entry in manifest["runs"]:
    if entry["track"] not in models:
        track = Circuit.load(entry["track"])
        models[entry["track"]] = FrenetModel(
            track.s, None, KinematicModelParameters(), p_kappa_np=track.curvature
        )
    model = models[entry["track"]]
    run = json.loads(gzip.decompress((root / "docs/data" / entry["file"]).read_bytes()))
    states = np.asarray(run["states"])
    indices = np.unique(np.linspace(0, len(states) - 2, 10).astype(int))
    for k in indices:
        for car in range(6):
            command = np.asarray(run["frames"][k]["commands"][car])
            result = solve_ivp(
                lambda t, x: np.asarray(model.f_ode(x, command)).ravel(),
                [0, run["dt"]],
                states[k, car],
                rtol=1e-9,
                atol=1e-10,
                max_step=0.02,
            )
            error = float(np.max(abs(result.y[:, -1] - states[k + 1, car])))
            worst = max(error, worst)
            assert error < 0.01, (entry["id"], k, car, error)
    print(f"Audited {entry['id']}: {len(indices) * 6} applied commands")
print(f"Maximum state discrepancy against adaptive integration: {worst:.3g} (mixed state units)")
