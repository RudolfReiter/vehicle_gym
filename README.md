# Vehicle Gym
This environment collects functions that are often needed when vehicle simulations are used. It contains essential solvers, vehicle models, and Frenet transformation tools. The following animation shows a simulation of five competing vehicles with different set speeds.

If you use this repository for your research, please cite:
```
@inproceedings{reiter_hierarchical_2023,
	title = {A Hierarchical Approach for Strategic Motion Planning in Autonomous Racing},
	doi = {10.23919/ECC57647.2023.10178143},
	booktitle = {European Control Conference ({ECC})},
	author = {Reiter, Rudolf and Hoffmann, Jasper and Boedecker, Joschka and Diehl, Moritz},
	month = jun,
	year = {2023},
	pages = {1--8},
}
```

![til](./animation1.gif)

## Authors
- [@RudolfReiter](https://www.github.com/RudolfReiter)

## Installation
Clone project
```
git clone https://github.com/RudolfReiter/vehicle_gym.git
```

Create virtual environment
```
python3 -m venv venv
source venv/bin/activate
```

Install requirements 
```
pip3 install -r requirements.txt
pip3 install -e .
```

### acados
First, install acados according to the [acados installation instructions](https://docs.acados.org/installation/index.html). 
Make sure to install the acados-Python interface accoring to [acados interface setup](https://docs.acados.org/python_interface/index.html):
```
pip install -e <acados_dir>/interfaces/acados_template
```
Remember to set external paths. In PyCharm, this can be done under "edit configurations" -> "environment variables":
- ACADOS_SOURCE_DIR=/dir/to/acados (e.g.,: /home/rudolf/acados)
- LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/dir/to/acados/lib (e.g.: /home/rudolf/acados/lib)

### acados beta version (optional)
For the globalization scripts, install the preliminary acados version from [David Kiessling's repository](https://github.com/david0oo/acados/tree/byrd_omojokun_development):

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
python tests/simulator/racetrack_cr.py
```
Globalized Fast Trajectory Planner (G-FAT) (Requires acados beta version)
```
python tests/planner_tests/globalization/racetracks.py
```

## Examples
The following animation shows part of a simulated race on the Spielberg race track involving three MPC competing agents with various strengths. Predictions, as seen in the ego vehicle of other race cars, are plotted in red. The planned ego trajectory is plotted in a color map from blue to green, where the color corresponds to the predicted time.
![til](./animation4.gif)
Let's examine other simulations involving five competing vehicles identical to the simulation above but with different road parameters.
![til](./animation2.gif)
![til](./animation3.gif)
