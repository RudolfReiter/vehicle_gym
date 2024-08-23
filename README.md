
# Vehicle Gym
This environment collects functions that are often needed when vehicle simulations are used. It contains basic solvers, vehicle models, and Frenet transformation tools.

![til](./animation1.gif)

## Authors
- [@RudolfReiter](https://www.github.com/RudolfReiter)


## Installation
```
pip3 install -r requirements.txt
pip3 install -e .
```

### acados
First, install the [acados core](https://docs.acados.org/installation/index.html). 
Make sure to install the acados-Python interface:
```
pip install -e <acados_dir>/interfaces/acados_template
```
Remember to set external paths. In PyCharm this can be done under "edit configurations" -> "environment variables":
- ACADOS_SOURCE_DIR=/dir/to/acados (e.g.,: /home/rudolf/acados)
- LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/dir/to/acados/lib (e.g.: /home/rudolf/acados/lib)


### Tests
Multi-agent tests on figure eight track, race track, or random track:
```
python tests/planner_tests/multi_agent/figure_eight.py
python tests/planner_tests/multi_agent/racetrack.py
python tests/planner_tests/multi_agent/random_track.py
```
Single-agent tests on a race track:
```
python tests/planner_tests/single_agent/racetrack.py
```
CommonRoad Simulator on a random race track:
```
python tests/planner_tests/simulator/racetrack.py
```
