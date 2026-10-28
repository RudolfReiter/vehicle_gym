# vehicle_gym

**Vehicle dynamics, nonlinear predictive planning, and reproducible driving scenes.**

[Research page](https://rudolfreiter.github.io/vehicle_gym/) · [ECC 2023 paper](https://ieeexplore.ieee.org/document/10178143) · [Author manuscript](https://publications.syscop.de/Reiter2023.pdf) · [HILEPP wrapper](https://github.com/RudolfReiter/hilepp)

![Actual traffic, the current ego plan, and opponent obstacle assumptions](docs/assets/driving-scene.png)

This branch improves Rudolf Reiter’s original repository and preserves its history. The portable planner and recorded experiments are new implementations, not a reproduction of the paper’s full benchmark. The original acados backend remains available.

## Research

**A Hierarchical Approach for Strategic Motion Planning in Autonomous Racing**<br>
Rudolf Reiter, Jasper Hoffmann, Joschka Boedecker, Moritz Diehl.<br>
European Control Conference (**ECC 2023**), Bucharest, pp. 1–8.<br>
DOI: [10.23919/ECC57647.2023.10178143](https://doi.org/10.23919/ECC57647.2023.10178143).

The research separates strategic references from constrained motion planning. This repository supplies the vehicle dynamics, reference interface, nonlinear planner, and seeded simulation. The separate HILEPP repository adds the learned high-level policy.

## Quick start

Python 3.9–3.12; the portable path requires no acados compilation.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[visualization,dev]'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m vehiclegym.demo
python -m pytest
python -m http.server 8000 --directory docs
```

Open <http://localhost:8000>. Committed replay data make the page usable immediately, even without running Python. The live page URL becomes available after GitHub Pages is enabled and this branch is merged.

```python
from vehiclegym.planner.references import FrenetReference
from vehiclegym.simulator.racing_world import RacingWorld

world = RacingWorld(max_steps=96)
observation = world.reset(seed=101, scenario="overtaking")
while not world.finished:
    observation, reward, terminated, truncated, info = world.step(
        FrenetReference(speed=32, lateral=3)
    )
    if info["solver_failed"]:
        print(info["plan"].status)
```

A failed optimizer solution ends the episode **before applying its control**. Returned controls are longitudinal force in newtons and steering rate in rad/s. The solver internally scales force to kN.

## What changed

- Complete `src/` package discovery and bundled track data, with a modern `pyproject.toml` and optional dependency groups.
- Lazy public imports: using the portable model no longer requires acados or plotting libraries.
- Independent planning snapshots and callable dataclass defaults; reference checks validate full dimensions and raise errors.
- Rectangle collision detection uses the separating-axis theorem, including crossing edges without contained vertices.
- Correct module names `predictor` and `evaluation_tools`, with compatibility imports for the old spellings.
- Executable examples moved out of the test suite; original GIFs moved to `docs/assets/legacy/`.
- A white research page with a continuous follow camera, current MPC horizon, opponent-assumption inspection, event markers, actual/reference traces, and downloadable figures/data.

See [migration notes](docs/MIGRATION.md) and [original project notes](docs/LEGACY.md).

## Layout

```text
src/vehicle_models/            Original vehicle models
src/vehiclegym/planner/        Original acados + new portable IPOPT planner
src/vehiclegym/road/track.py    Lightweight geometry and packaged CSV tracks
src/vehiclegym/simulator/      Original simulators + seeded RacingWorld
examples/                     Original acados demonstrations
tests/unit/                  Regression and numerical tests
docs/                         Static research page and recorded results
scripts/                      Figure generation, validation, source export
```

## Numerical setup and limits

The portable backend uses the original five-state Frenet bicycle model `[s, n, alpha, v, delta]`, RK4 integration, 12 intervals of 0.25 s, and nonlinear multiple shooting in CasADi/IPOPT. It constrains road position, speed, steering angle/rate, longitudinal force, lateral acceleration, and obstacle ellipses. Obstacle slack is bounded to `[0, 0.08]` and penalized; solver residuals and slack are exported separately.

MPC **replans every 0.25 s and applies only the first control**. Opponent trajectories are regenerated from current measurements in every solve: constant speed and measured heading for leading/alongside cars; full braking for followers as a yielding feasibility assumption. The displayed orange trajectories are exactly those used by the optimizer. See [control and prediction semantics](docs/CONTROL_AND_PREDICTION.md).

Opponents actively seek passing gaps and slow for traffic. They are not the paper’s MPC opponent policies. Blocking challengers target 34–36 m/s while the ego is limited to 32 m/s, so simply accelerating cannot escape them. Collision checks use oriented Cartesian rectangles every 0.05 s. Obstacle ellipses remain in Frenet coordinates. No terminal invariant set is implemented, and there is no recursive-feasibility or safety guarantee.

The nine recorded runs compare three references across three tasks using seed 101 and 24-second episodes. All nine completed without a detected collision or solver failure. In blocking, the center-reference controller loses two places; this is a fixed-reference baseline, not learned defense. See [recorded metrics](docs/data/experiment.json).

Packaged real track geometry is also available:

```python
from vehiclegym.road.track import Track
print(Track.available_tracks())
track = Track.from_csv("Spielberg")
```

The published portable evaluation uses seeded synthetic circuit sections; it does not claim a full-lap real-track benchmark.

## Original acados backend

Install `pip install -e '.[legacy]'`, then install/build acados and its Python interface as described in the [upstream installation documentation](https://docs.acados.org/installation/index.html). The older dependency list is preserved in `requirements-legacy.txt`. The original demonstrations are in `examples/`. Their compiled acados execution has not been validated in this update. Portable tests exercise the original dynamics but not every legacy planner or simulator.

## Rebuild the visual artifacts

```bash
python scripts/build_report.py
node scripts/render_scene.mjs
# Optional PNG export: requires system GI / Rsvg / Cairo / Pillow
/usr/bin/python3 scripts/rasterize_scenes.py
python scripts/build_site.py
node scripts/check_interactions.mjs
python scripts/check_data.py
```

The renderer export records the same Canvas drawing commands used by the page. It is not a browser screenshot. No external fonts or JavaScript libraries are needed. The page starts paused, never loops automatically, and pauses when the tab is hidden.

## Validation and provenance

Original source: upstream commit `09791bf` (full hash in `PROVENANCE.json`). Local verification uses Python 3.9.23, NumPy 1.26.4, SciPy 1.13.1, CasADi 3.8.1. CI also targets Python 3.11. Runtime-dependent solve times are included as diagnostics, not real-time guarantees. See `PROVENANCE.json` for versions and data hashes.

MIT license, consistent with the original package metadata. Cite the paper through [CITATION.cff](CITATION.cff).
