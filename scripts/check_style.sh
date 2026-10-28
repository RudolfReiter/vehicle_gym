#!/usr/bin/env bash
set -euo pipefail
# Deliberately check the new portable API; preserve upstream legacy formatting.
ruff check src/vehiclegym/__init__.py src/vehiclegym/demo.py src/vehiclegym/planner/portable.py src/vehiclegym/planner/opponent_prediction.py src/vehiclegym/planner/references.py src/vehiclegym/road/track.py src/vehiclegym/simulator/racing_world.py tests/unit scripts
ruff format --check src/vehiclegym/__init__.py src/vehiclegym/demo.py src/vehiclegym/planner/portable.py src/vehiclegym/planner/opponent_prediction.py src/vehiclegym/planner/references.py src/vehiclegym/road/track.py src/vehiclegym/simulator/racing_world.py tests/unit scripts
