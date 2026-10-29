# vehicle_gym — multi-agent model predictive racing

Six independent nonlinear MPC controllers compete on packaged racing circuits. Every controller observes the same measured race state, predicts its five opponents, solves a new optimal control problem, and applies only the first input. The simulation then advances all cars together by 0.2 s.

[Interactive races](https://rudolfreiter.github.io/vehicle_gym/) · [ECC 2023 paper](https://publications.syscop.de/Reiter2023.pdf) · [HILEPP comparison](https://github.com/RudolfReiter/hilepp)

![Recorded six-car race](docs/assets/driving-scene.png)

The research interface presents continuous 180-second IMS races and a Sepang stress test. Select any car to inspect its current MPC horizon and its own assumptions about the other five vehicles. Race order, measured speed, numerical diagnostics and all published outcomes accompany the replay. Playback interpolates measured states and never loops automatically.

There is no RL policy or scripted blocking scenario in this experiment. Different target speeds and lateral references produce the competition. The separate HILEPP repository compares HILEPP-II, direct SAC and pure MPC against the same five MPC opponents.

## Run the experiment

Python 3.9+, a C compiler, CMake and a pinned acados installation are required for the new racing backend. The original portable IPOPT examples remain available.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,visualization]'
bash scripts/setup_acados.sh
# Export the ACADOS_SOURCE_DIR and LD_LIBRARY_PATH printed by the script.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m vehiclegym.race_demo --tracks IMS Sepang --duration 180 \
  --seeds 20000 20001 20002 --output docs/data
python scripts/check_data.py
python scripts/build_report.py
python -m http.server 8000 --directory docs
```

Open http://localhost:8000. The included compressed replays work without Python, acados or a training framework when served by GitHub Pages. Opening `index.html` directly with a `file:` URL does not permit replay fetching in most browsers.

## Research implementation

- **Vehicle model:** original kinematic Frenet dynamics, states `[s, n, alpha, v, delta]`, controls longitudinal force and steering rate. Actual motion uses RK4 with four substeps per applied control.
- **Independent MPC:** six solver instances and warm starts, 25 shooting intervals over 5 s, acados SQP. Multiple initial guesses are tried when the first solve has excessive obstacle slack or numerical residual. Every candidate solves the same OCP; no heuristic actuator fallback is used.
- **Interaction model:** measured speed and heading error remain constant for leading/alongside cars in curvilinear coordinates. A car more than 4.6 m behind is assumed to brake at 10 m/s² until standstill. These are optimization assumptions, not forecasts of the other MPCs' intentions. The controller never reads their future plans.
- **Constraints:** soft state, lateral-acceleration and elliptical obstacle constraints; hard actuator bounds; approximate terminal heading and speed conditions. Slack and solver failures are recorded. This recreation does not establish recursive feasibility or reproduce the paper's quantitative results.
- **Circuits:** packaged IMS and Sepang coordinates, periodic arc-length parameterization, 6 m coordinate smoothing. Road constraints conservatively use the minimum packaged half-width; the renderer shows the same conservative usable width as the simulation. Surveyed widths remain in the exported track data.
- **Evaluation:** three fixed test seeds per circuit, matched starts across policies in HILEPP. A collision, track exit or rejected solve by any vehicle ends the entire race and remains in the published results. An opponent failure is identified by vehicle number.

See [control and prediction semantics](docs/CONTROL_AND_PREDICTION.md) and the [migration guide](docs/MIGRATION.md). Upstream examples, models and their [original documentation](docs/LEGACY.md) remain available.

## Source layout

| Module | Responsibility |
|---|---|
| `road/circuit.py` | Periodic packaged circuit geometry and nearest-lap opponent positions |
| `planner/race_mpc.py` | Independent acados OCP, constraints, warm starts and numerical acceptance |
| `planner/opponent_prediction.py` | Measured-state opponent assumptions, integrated with RK4 |
| `simulator/circuit_race.py` | Synchronous six-car race and physical collision checks |
| `race_replay.py` | Recorded states, commands, all six plans and evaluation metrics |
| `docs/` | Static scientific viewer; current plans, race timing and telemetry |

## Verification

```bash
python -m pytest
bash scripts/check_style.sh
python scripts/check_data.py
node scripts/check_interactions.mjs
node scripts/render_scene.mjs
python scripts/build_site.py
```

Unit tests check shared-state replanning, direct-SAC bypass, HILEPP parameter isolation, circuit seams and opponent-mode changes. Data checks inspect every published controller frame. The JavaScript checks use DOM and Canvas doubles and do not substitute for browser layout testing. Numerical race generation requires acados; CI can validate committed replays without it.

## Citation

Rudolf Reiter, Jasper Hoffmann, Joschka Boedecker, and Moritz Diehl. **A Hierarchical Approach for Strategic Motion Planning in Autonomous Racing.** European Control Conference (ECC), 2023. [DOI: 10.23919/ECC57647.2023.10178143](https://doi.org/10.23919/ECC57647.2023.10178143).
