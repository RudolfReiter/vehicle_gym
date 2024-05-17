from copy import copy
from typing import List, TYPE_CHECKING
from dataclasses import dataclass
import numpy as np
import casadi as cs

if TYPE_CHECKING:
    from acados_template import AcadosOcp, AcadosOcpSolver


@dataclass
class AcadosParameter:
    name: str = None
    casadi_obj = None
    length: int = 0
    width: int = 0
    size: int = 0
    idx_start: int = 0


class ParamHandler:
    """
    Acados parameter handler
    """

    def __init__(self, acados_ocp: "AcadosOcp", num_nodes: int):
        self.ocp = acados_ocp
        self.solver = None
        self.par_list = []
        self.num_nodes = num_nodes
        self.idx_pointer = 0
        self.is_solver_list = False

    def set_solver(self, solver: "AcadosOcpSolver"):
        self.solver = solver
        self.is_solver_list = False

    def set_solver_list(self, solver: List["AcadosOcpSolver"]):
        self.solver = solver
        self.is_solver_list = True

    def add(self, name: str, length: int, width: int = 1):
        assert not self._check_existence(name), f"Parameter \"{name}\" exists already!"

        par_obj = AcadosParameter()
        par_obj.name = name
        par_obj.casadi_obj = cs.MX.sym(name, length, width)
        par_obj.length = length
        par_obj.width = width
        par_obj.size = length * width
        par_obj.idx_start = copy(self.idx_pointer)
        self.par_list.append(par_obj)

        self.idx_pointer += length * width

    def set_par(self, name: str, val: np.ndarray, stage: int = -1, stage_end: int = -1, i_element: int = 0):
        if stage < 0 and stage_end < 0:
            stage = 0 if stage < 0 else stage
            stage_end = self.num_nodes
        elif stage >= 0 and stage_end < 0:
            stage_end = stage + 1
        elif stage < 0 and stage_end >= 0:
            stage = 0
        assert stage <= stage_end, (f"Error, while setting parameter {name}. "
                                    f"Start index {stage} is greaten than end index {stage_end}!")
        assert self._check_existence(name), f"Parameter \"{name}\" does not exist!"
        assert stage <= self.num_nodes and stage_end <= self.num_nodes, "Index out of range!"

        par = [par for par in self.par_list if par.name == name][0]
        par_idxs = np.array([par.idx_start + i_element * par.length,
                             par.idx_start + (i_element + 1) * par.length])
        par_idxs = np.array(list(range(par_idxs[0], par_idxs[1])))
        assert val.shape[0] == par.length, f"Wrong length {val.shape[0]} for parameter {name} with length {par.length}!"
        assert par.size >= (i_element + 1) * par.length, f"Out of range value of parameter element!"

        if self.is_solver_list:
            for stage in range(stage, stage_end):
                [solver.set_params_sparse(stage, par_idxs, val) for solver in self.solver]
        else:
            for stage in range(stage, stage_end):
                self.solver.set_params_sparse(stage, par_idxs, val)

    def get_all_casadi(self):
        return cs.vertcat(*[cs.vec(par.casadi_obj) for par in self.par_list])

    def get_element(self, name: str):
        return [par for par in self.par_list if par.name == name][0]

    def get_casadi(self, name, i_element: int = -1):
        assert self._check_existence(name), f"Parameter \"{name}\" does not exist!"
        par = [par for par in self.par_list if par.name == name][0]
        if par.width > 1 and i_element >= 0:
            return par.casadi_obj[:, i_element]
        else:
            return par.casadi_obj

    def get_len(self) -> int:
        return copy(self.idx_pointer)

    def _check_existence(self, name: str) -> bool:
        if name in [par.name for par in self.par_list]:
            return True
        else:
            return False
