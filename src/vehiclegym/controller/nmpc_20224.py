import math
from copy import copy
from dataclasses import dataclass
import casadi as cs
import numpy as np
import scipy
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from matplotlib import pyplot as plt
from scipy import interpolate
from vehiclegym.road.road import RoadOptions
from vehicle_models.model_kinematic import KinematicModelParameters
from vehicle_models.model_kinematic_cartesian import KinematicModelCartesianSmall


@dataclass
class NmpcOptions20224:
    n_nodes: int = 30  # Time steps for optimization horizon
    t_prediction: float = 3
    weight_inputs: np.ndarray = np.array([1e-5, 5e5])
    weight_states: np.ndarray = np.array([1e1, 1e1, 1e-4, 0])
    weight_states_end: np.ndarray = np.array([1e0, 1e0, 1e-1, 0])

    def get_sampling_time(self) -> float:
        return self.t_prediction / self.n_nodes


class Nmpc20224:
    def __init__(self,
                 model_vehicle: KinematicModelCartesianSmall,
                 planner_options: NmpcOptions20224,
                 x0: np.ndarray = np.array([0, 0, 0, 0])):
        # create render arguments
        self.planner_opts = planner_options
        ocp = AcadosOcp()

        # set model
        model, constraint_expr = model_vehicle.export4acados()
        self.model = model
        ocp.model = model

        # define constraint
        ocp.model.con_h_expr = constraint_expr

        # set dimensions
        nx = model.x.size()[0]
        self.nx = nx
        nu = model.u.size()[0]
        self.nu = nu
        ny = nx + nu
        ny_e = nx

        ocp.dims.N = planner_options.n_nodes
        ocp.solver_options.tf = planner_options.t_prediction
        td = ocp.solver_options.tf / ocp.dims.N

        ns = 3  # PROBABLY all slacks together?
        nsh = 3

        # set cost
        Q = np.diag(planner_options.weight_states) * td
        R = np.diag(planner_options.weight_inputs) * td
        Qe = np.diag(planner_options.weight_states_end)

        ocp.cost.cost_type = "LINEAR_LS"
        ocp.cost.cost_type_e = "LINEAR_LS"

        ocp.cost.W = scipy.linalg.block_diag(Q, R)
        ocp.cost.W_e = Qe

        Vx = np.zeros((ny, nx))
        Vx[:nx, :nx] = np.eye(nx)
        ocp.cost.Vx = Vx

        Vu = np.zeros((ny, nu))
        Vu[4, 0] = 1.0
        Vu[5, 1] = 1.0
        ocp.cost.Vu = Vu

        Vx_e = np.zeros((ny_e, nx))
        Vx_e[:nx, :nx] = np.eye(nx)
        ocp.cost.Vx_e = Vx_e

        ocp.cost.zl = np.array([0,  1e5, 1e5])
        ocp.cost.Zl = np.array([1e5, 0, 0])
        ocp.cost.zu = ocp.cost.zl
        ocp.cost.Zu = ocp.cost.Zl

        # set intial references
        ocp.cost.yref = np.array([0, 0, 0, 0, 0, 0])
        ocp.cost.yref_e = np.array([0, 0, 0, 0])

        # setting constraints
        ocp.constraints.lbx = np.array([0])
        ocp.constraints.ubx = np.array([model_vehicle.params_.maximum_velocity])
        ocp.constraints.idxbx = np.array([3])

        ocp.constraints.lbu = np.array([-model_vehicle.params_.maximum_deceleration_force,
                                        -model_vehicle.params_.maximum_steering_angle])
        ocp.constraints.ubu = np.array([model_vehicle.params_.maximum_acceleration_force,
                                        model_vehicle.params_.maximum_steering_angle])
        ocp.constraints.idxbu = np.array([0, 1])

        ocp.constraints.lh = np.array([-model_vehicle.params_.maximum_lateral_acc, -1e6, -1e6])
        ocp.constraints.uh = np.array([model_vehicle.params_.maximum_lateral_acc, 1e6, 1e6])

        ocp.constraints.lsh = np.zeros(nsh)
        ocp.constraints.ush = np.zeros(nsh)
        ocp.constraints.idxsh = np.array([0, 1, 2])

        # set initial condition
        ocp.constraints.x0 = x0

        # ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.sim_method_num_stages = 2
        ocp.solver_options.sim_method_num_steps = 1

        # ocp.solver_options.qp_solver_tol_stat = 1e-2
        # ocp.solver_options.qp_solver_tol_eq = 1e-2
        # ocp.solver_options.qp_solver_tol_ineq = 1e-2
        # ocp.solver_options.qp_solver_tol_comp = 1e-2

        # create solver
        self.solver = AcadosOcpSolver(ocp, json_file="acados_ocp.json")
        self.acados_integrator = AcadosSimSolver(ocp)

    def step(self, x_measured: np.ndarray, y_ref: np.ndarray) -> [np.ndarray, int]:
        self.solver.set(0, "lbx", x_measured)
        self.solver.set(0, "ubx", x_measured)
        terminal_constraint_l = np.array([-8,
                                          y_ref[0, self.planner_opts.n_nodes],
                                          y_ref[1, self.planner_opts.n_nodes]])
        terminal_constraint_u = np.array([8,
                                          y_ref[0, self.planner_opts.n_nodes],
                                          y_ref[1, self.planner_opts.n_nodes]])
        self.solver.constraints_set(self.planner_opts.n_nodes-1, "lh", terminal_constraint_l)
        self.solver.constraints_set(self.planner_opts.n_nodes-1, "uh", terminal_constraint_u)

        # set reference for stages that include u
        for stage in range(self.planner_opts.n_nodes):
            self.solver.cost_set(stage, "yref", y_ref[:, stage])

        # set final reference without u
        yref = y_ref[:-2, self.planner_opts.n_nodes]
        self.solver.cost_set(self.planner_opts.n_nodes, "yref", yref)

        status = self.solver.solve()

        if status != 0:
            raise Exception('acados acados_ocp_solver returned status {}. Exiting.'.format(status))

        u = self.solver.get(0, "u")

        return u, status


if __name__ == "__main__":
    print("-------------------------")
    print("Test trajectory controller")
    print("-------------------------")
    params = KinematicModelParameters()
    model = KinematicModelCartesianSmall(params)
    controller_options = NmpcOptions20224()

    x0 = np.array([0, 0, 0, 0])
    controller = Nmpc20224(model_vehicle=model, planner_options=controller_options, x0=x0)

    Nsim = 1300
    simX = np.ndarray((Nsim + 1, controller.nx))
    simU = np.ndarray((Nsim, controller.nu))

    xcurrent = x0
    simX[0, :] = xcurrent
    n_nodes = controller.planner_opts.n_nodes

    # create reference
    td = controller_options.get_sampling_time()
    t_ref = np.arange(0, Nsim * td + 10, td)
    phi_ref = t_ref * 2 * np.pi / 50 - np.pi / 2
    r_ref = 100 - 0.5 * t_ref
    pos_x_ref = r_ref * np.cos(phi_ref)
    pos_y_ref = r_ref * np.sin(phi_ref) + 100
    theta_ref = phi_ref + np.pi / 2

    y_ref = np.zeros((4 + 2, n_nodes + 1))

    # closed loop
    for i in range(Nsim):

        # set reference
        for stage in range(controller.planner_opts.n_nodes + 1):
            y_ref[:, stage] = np.array([pos_x_ref[i + stage], pos_y_ref[i + stage], theta_ref[i + stage], 10, 0, 0])

        # step controller
        simU[i, :], status_controller = controller.step(xcurrent, y_ref)

        # simulate system
        controller.acados_integrator.set("x", xcurrent)
        controller.acados_integrator.set("u", simU[i, :])

        status = controller.acados_integrator.solve()
        if status != 0:
            raise Exception('acados integrator returned status {}. Exiting.'.format(status))

        # update state
        xcurrent = controller.acados_integrator.get("x")
        simX[i + 1, :] = xcurrent

    t_sim = np.arange(0, Nsim * td + 1e-6, td)
    fig, axs = plt.subplots(4, 1)

    axs[0].plot(pos_x_ref, pos_y_ref, label="reference", color="r")
    axs[0].plot(simX[:, 0], simX[:, 1], label="position")
    axs[0].legend()
    axs[0].axis("equal")
    axs[1].plot(t_sim, simX[:, 3], label="speed")
    axs[1].legend()
    axs[2].plot(t_sim[:-1], simU[:, 0], label="force")
    axs[2].legend()
    axs[3].plot(t_sim[:-1], simU[:, 1], label="steering angle")
    axs[3].legend()
    plt.show()
