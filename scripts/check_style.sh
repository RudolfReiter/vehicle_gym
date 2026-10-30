#!/usr/bin/env bash
set -euo pipefail
# Check the maintained APIs; preserve upstream legacy formatting.
files=(src/vehiclegym/__init__.py src/vehiclegym/demo.py
  src/vehiclegym/race_demo.py src/vehiclegym/race_replay.py
  src/vehiclegym/planner/portable.py src/vehiclegym/planner/race_mpc.py
  src/vehiclegym/planner/opponent_prediction.py src/vehiclegym/planner/references.py
  src/vehiclegym/planner/vehicle_limits.py
  src/vehiclegym/road/track.py src/vehiclegym/road/circuit.py
  src/vehiclegym/simulator/racing_world.py src/vehiclegym/simulator/circuit_race.py
  tests/unit tests/test_circuit_race.py scripts)
ruff check "${files[@]}"
ruff format --check "${files[@]}"
