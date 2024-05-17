from copy import deepcopy
from typing import Optional, Type

import numpy as np
from dataclasses import dataclass
from scipy import interpolate


@dataclass
class CartesianTrajectory:
    x: np.ndarray = np.zeros((1,))
    y: np.ndarray = np.zeros((1,))
    phi: np.ndarray = np.zeros((1,))
    v: np.ndarray = np.zeros((1,))
    delta: np.ndarray = np.zeros((1,))

    def __init__(self, trajectory: np.ndarray = None):
        if trajectory is not None:
            self.set_as_array(trajectory)

    def get_as_array(self):
        return np.vstack((self.x, self.y, self.phi, self.v, self.delta))

    def set_as_array(self, trajectory: np.ndarray):
        if len(trajectory.shape) == 2:
            self.x = trajectory[0, :]
            self.y = trajectory[1, :]
            self.phi = trajectory[2, :]
            self.v = trajectory[3, :]
            self.delta = trajectory[4, :]
        elif len(trajectory.shape) == 1:
            self.x = np.array([trajectory[0]])
            self.y = np.array([trajectory[1]])
            self.phi = np.array([trajectory[2]])
            self.v = np.array([trajectory[3]])
            self.delta = np.array([trajectory[4]])
        else:
            assert False


@dataclass
class FrenetTrajectory:
    s: np.ndarray = np.zeros((1,))
    n: np.ndarray = np.zeros((1,))
    alpha: np.ndarray = np.zeros((1,))
    v: np.ndarray = np.zeros((1,))
    delta: np.ndarray = np.zeros((1,))
    t: np.ndarray = np.zeros((1,))

    def __init__(self, trajectory: np.ndarray = None, t: np.ndarray = np.zeros((1,))):
        if trajectory is not None:
            self.set_as_array(trajectory)
        self.t = t

    def __add__(self, other: np.ndarray):
        if len(other.shape) == 2:
            assert other.shape[1] == self.s.shape[0]
        elif len(other.shape) == 1:
            assert other.shape[0] == 5
        else:
            raise NotImplementedError
        return_class = deepcopy(self)
        return_class.s += other[0, ...]
        return_class.n += other[1, ...]
        return_class.alpha += other[2, ...]
        return_class.v += other[3, ...]
        return_class.delta += other[4, ...]
        return return_class

    def get_as_array(self):
        return np.vstack((self.s, self.n, self.alpha, self.v, self.delta))

    def set_as_array(self, trajectory: np.ndarray):
        if len(trajectory.shape) == 2:
            self.s = trajectory[0, :]
            self.n = trajectory[1, :]
            self.alpha = trajectory[2, :]
            self.v = trajectory[3, :]
            self.delta = trajectory[4, :]
        elif len(trajectory.shape) == 3:
            self.s = trajectory[0, :, 0]
            self.n = trajectory[1, :, 0]
            self.alpha = trajectory[2, :, 0]
            self.v = trajectory[3, :, 0]
            self.delta = trajectory[4, :, 0]
        elif len(trajectory.shape) == 1:
            self.s = np.array([trajectory[0]])
            self.n = np.array([trajectory[1]])
            self.alpha = np.array([trajectory[2]])
            self.v = np.array([trajectory[3]])
            self.delta = np.array([trajectory[4]])
        else:
            assert False

    def resample(self, delta_t: float):
        if self.t.shape[0] < 2:
            assert False
        x = self.get_as_array()
        interp_fun_t2x = interpolate.interp1d(self.t, x, kind="linear", axis=1, fill_value='extrapolate')
        t_new = np.arange(start=self.t[0], stop=self.t[-1] + 1e-9, step=delta_t)
        x_resampled = interp_fun_t2x(t_new)
        self.set_as_array(x_resampled)
        self.t = t_new

    def add_time_axis(self, delta_t: float):
        self.t = np.arange(start=0, stop=delta_t * self.s.shape[0] + 1e-9, step=delta_t)

    def add_f_sample(self, state: "FrenetTrajectory", delta_t: float = None, t: float = None):
        state_vec = self.get_as_array()
        state_vec = np.append(state_vec, np.array([state.s, state.n, state.alpha, state.v, state.delta]), axis=1)
        self.set_as_array(state_vec)
        if delta_t is not None and t is None:
            self.t = np.append(self.t, self.t[-1] + delta_t)
        elif delta_t is None and t is not None:
            self.t = np.append(self.t, t)
        elif delta_t is not None and t is not None:
            assert False

    def add_np_sample(self, new_states: np.ndarray, t: np.ndarray = None):
        state_vec = self.get_as_array()
        if len(state_vec.shape) < 2:
            state_vec = np.expand_dims(state_vec, axis=0)

        if len(new_states.shape) < 2:
            new_states = np.expand_dims(new_states, axis=0)
        if new_states.shape[0] != 5:
            new_states = new_states.transpose()
        state_vec = np.append(state_vec, new_states, axis=1)
        self.set_as_array(state_vec)

        if t is not None:
            self.t = np.append(self.t, t)

    def get_last_state(self) -> [np.ndarray, float]:
        return np.array([self.s[-1], self.n[-1], self.alpha[-1], self.v[-1], self.delta[-1]]), self.t[-1]


if __name__ == "__main__":
    traj_f = FrenetTrajectory(np.array([0, 2, 3, 4, 3]))
    traj_f.add_np_sample(np.array([[90, 2, 3, 4, 3], [10, 2, 3, 4, 3]]), t=np.array([2, 3]))
    traj_f.add_np_sample(np.array([100, 2, 3, 4, 3]), t=np.array([10]))

    print(traj_f.s)
    print(traj_f.get_last_state())
    traj_f.resample(delta_t=1)
    print(traj_f.s)
    print(traj_f.t)
