""" Homotopy selector module for the roborace framework

export ACADOS_SOURCE_DIR="/home/rudolf/Programs/acados"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"/home/rudolf/Programs/acados/lib"
source $ARG_WS_DIR/config/extensions.sh

This module starts a ROS node that receives obstacle and road information and returns a smooth center line, that avoids
the obstacles.

Autor: Rudolf Reiter
Date: Dez. 2022
"""

from typing import List
import rclpy
from rclpy.node import Node

import sys
import numpy as np

import matplotlib
from scipy.signal import savgol_filter
from matplotlib import pyplot as plt
from copy import deepcopy, copy
from threading import Lock
import time
from nav_msgs.msg import Odometry
from arg_msgs.msg import Path, PathPoint, PositionOnTrack, Trajectory
from arg_msgs.msg import ObjectList, Object, VehicleLimits

from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
from vehiclegym.automotive_datastructures import FrenetTrajectory
from vehiclegym.ros_conversion_tools import (
    get_ros_data_vehicle_state,
    get_ros_waypoints,
)
from vehicle_models.model_kinematic import KinematicModelParameters
from vehiclegym.trajectory_planner_acados_bilevel import VehiclePlannerAcados2022Bilevel
from vehiclegym.trajectory_planner_acados_20221 import (
    PlannerOptions,
)
from vehiclegym.helpers import json2dataclass
from vehiclegym.road import Road, RoadOptions


class RosPlanner:
    def __init__(self) -> None:
        # Define ROS attributes
        self.odometry = None
        self.path = None
        self.position_on_track = None
        self.object_list = None
        self.vehicle_limits = None


class MPCRL(Node):
    def __init__(self):
        super().__init__("arg_mpcrl")

        # ROS options
        self.declare_parameter("node_rate", 10)
        self.node_rate = (
            self.get_parameter("node_rate").get_parameter_value().integer_value
        )

        self.declare_parameter("x", 1234.56)
        self.x = self.get_parameter("x").get_parameter_value().double_value

        # temp
        self.did_update = False

        # Define ROS attributes
        self.odometry = None
        self.path = None
        self.position_on_track = None
        self.object_list = None
        self.vehicle_limits = None

        self.is_initialized = False
        self._track_len = 3186

        self.diagnostics = DiagnosticArray()

        self.time_old = self.get_clock().now()

        # Locks for safe data transfer (Python, multi-threading: You do not know what's going on!)
        self.odo_lock = Lock()
        self.path_lock = Lock()
        self.position_on_track_lock = Lock()
        self.object_lock = Lock()
        self.vehicle_limits_lock = Lock()

        # Publisher
        self.pub_trajectory = self.create_publisher(
            Trajectory, "devbot/mppoav/trajectory", 10
        )
        self.pub_diagnostics = self.create_publisher(
            DiagnosticArray, "diagnostics/mpcrl", 10
        )

        # Subscriber
        self.sub_odometry = self.create_subscription(
            Odometry, "devbot/odom", self.odometry_callback, 10
        )
        self.sub_path = self.create_subscription(
            Path, "devbot/map/path", self.path_callback, 10
        )  # Path is part of the Track!
        self.sub_position_on_track = self.create_subscription(
            PositionOnTrack,
            "devbot/position_on_track",
            self.position_on_track_callback,
            10,
        )
        self.sub_object_list = self.create_subscription(
            ObjectList, "devbot/objects/prediction", self.object_list_callback, 10
        )
        # self.sub_track = self.create_subscription(
        #    VehicleLimits,
        #    "devbot/behavior_planner/vehicle_limits",
        #    self.vehicle_limits_callback,
        #    10,
        # )

        self.timer = self.create_timer(1.0 / self.node_rate, self.timer_callback)

        # initialize planner
        model_path = "benchmarks/model/"
        planner_path = "benchmarks/planner/"
        # Load parameters
        self.ego_model_params = json2dataclass(
            KinematicModelParameters, relpath=model_path, filename="devbot.json"
        )

        self.planner_options = json2dataclass(
            PlannerOptions, relpath=planner_path, filename="main_devbot.json"
        )

        self.default_action = np.array(self.planner_options.actions_default)
        self.get_logger().info("MPCRL - Planner Started")
        self.time_now = 0

    def messages_received(self) -> bool:
        """Checks if all required ros messages were received

        Returns:
            bool: True, if all messages received
        """
        msgs2check = [
            self.odometry,
            self.path,
            self.position_on_track,
            self.object_list,
        ]
        received = not any(v is None for v in msgs2check)
        return received

    def timer_callback(self):
        """Main ROS callback"""
        time_now = self.get_clock().now()
        self.time_now = time_now.nanoseconds / 1e9

        self.diagnostics.header.stamp = time_now.to_msg()
        diagnostic_status = DiagnosticStatus()
        diagnostic_status.message = "OK"
        diagnostic_status.level = DiagnosticStatus.OK
        diagnostic_status.name = "arg_mpcrl"
        diagnostic_status.hardware_id = "123456"
        self.diagnostics.status.clear()
        self.diagnostics.status.append(diagnostic_status)
        self.pub_diagnostics.publish(self.diagnostics)

        if self.is_initialized:
            self.main_iteration()
        else:
            if self.messages_received():
                self.get_logger().info("MPCRL - Planner Initializing...")
                self.initialize()
                self.get_logger().info("MPCRL - Planner Initialized")
            else:
                self.get_logger().info("MPCRL - Planner Waiting for all messages...")
        # self.get_logger().info(
        #    "MPCRL - Planner: Solution time: {}".format((time_now - self.time_old))
        # )
        self.time_old = time_now

    def get_ego_state_f(self, road: Road = None) -> np.ndarray:
        """Obtain ego state in Frenet coordinates (s,n,alpha,v,delta) from ros message

        Returns:
            np.ndarray: Frenet state [s,n,alpha,v,delta]
        """
        # get cartesian state
        (
            devbot_x,
            devbot_y,
            devbot_z,
            devbot_psi,
            devbot_v,
            devbot_yaw_rate,
        ) = get_ros_data_vehicle_state(self.odometry)

        # which road to use for transformation
        if road is None:
            road = self.current_road

        s = self.position_on_track.s
        n = self.position_on_track.d

        if s < road.s_grid_[0]:
            s += self._track_len

        alpha = np.mod(devbot_psi - road.spline_s2phi_(s) + np.pi, np.pi * 2) - np.pi
        v = self.position_on_track.v
        delta = 0

        return np.array([s, n, alpha, v, delta])

    def get_opp_states_f(self) -> List[np.ndarray]:
        """Obtain opponent states in Frenet coordianted (s,n,alpha,v,delta)

        Returns:
            List[np.ndarray]: List of opponent states [s,n,alpha,v,delta]
        """
        pass

    def get_path_local(self) -> List[np.ndarray]:
        """Get local path from ros messages

        Returns:
            List[np.ndarray]: [xy, theta, s_grid, kappa_grid, nl_grid, nr_grid]
        """
        track_data, track_len, overlap = get_ros_waypoints(self.path)
        return track_data

    def send_trajectory(
        self, x_full_c: np.ndarray, x_full_f: np.ndarray, u_full: np.ndarray
    ) -> None:
        """Sends ros message with planned trajectory

        Args:
            x_full (np.ndarray): size of 5xN trajectory in local frenet coordinate frame
        """
        x_full_f = copy(x_full_f)
        x_full_f[0, :] = x_full_f[0, :] - x_full_f[0, 0]

        if True:
            M_add = 25
            ds = 1
            x_full_c = copy(x_full_c)
            x_full_f = copy(x_full_f)
            u_full = copy(u_full)

            dx_c0 = x_full_c[:, 1] - x_full_c[:, 0]
            dx_e = dx_c0[0:2] / np.sqrt(dx_c0[0] ** 2 + dx_c0[1] ** 2) * ds

            x_c0 = x_full_c[:, 0]
            x_f0 = x_full_f[:, 0]
            x_u0 = u_full[:, 0]

            x_full_c = np.hstack((np.zeros((5, M_add)), x_full_c))
            x_full_f = np.hstack((np.zeros((5, M_add)), x_full_f))
            u_full = np.hstack((np.zeros((2, M_add)), u_full))

            x_full_c[:, 0:M_add] = np.expand_dims(x_c0, 1)
            x_full_f[:, 0:M_add] = np.expand_dims(x_f0, 1)
            u_full[:, 0:M_add] = np.expand_dims(x_u0, 1)

            for j in range(0, M_add):
                i = M_add - j
                x_full_c[0:2, j] = x_c0[0:2] - i * dx_e
                x_full_f[0, j] = x_f0[0] - i * ds

        # Create ROS trajectory
        trajectory = Trajectory()
        trajectory.x = x_full_c[0, :].astype(np.float32).tolist()
        trajectory.y = x_full_c[1, :].astype(np.float32).tolist()
        trajectory.psi = x_full_c[2, :].astype(np.float32).tolist()
        trajectory.v = x_full_c[3, :].astype(np.float32).tolist()

        trajectory.s = x_full_f[0, :].astype(np.float32).tolist()
        trajectory.header.stamp = self.get_clock().now().to_msg()
        trajectory.header.frame_id = "map"
        trajectory.v_valid = True
        trajectory.a_valid = True

        trajectory.delta = x_full_c[4, :].astype(np.float32).tolist()
        trajectory.phi = x_full_c[2, :].astype(np.float32).tolist()
        trajectory.psi_frenet = x_full_f[2, :].astype(np.float32).tolist()

        trajectory.a = (
            (np.append(u_full[0, :] / self.ego_model_params.mass, 0.0))
            .astype(np.float32)
            .tolist()
        )
        trajectory.t = copy(self.planner.time_grid).tolist()

        trajectory.status = 1

        self.pub_trajectory.publish(copy(trajectory))

    def initialize(self):
        """Initialize the planner"""

        (
            xy_grid,
            theta_grid,
            s_grid,
            kappa_grid,
            nl_grid,
            nr_grid,
            v_max,
        ) = self.get_path_local()
        filter_coefs = (51, 5)
        kappa_grid = savgol_filter(kappa_grid, filter_coefs[0], filter_coefs[1])

        self.current_road = Road(
            RoadOptions(),
            kappa=kappa_grid,
            s=s_grid,
            p_xy=xy_grid,
            phi=theta_grid,
            nl=nl_grid,
            nr=nr_grid,
        )

        initial_state = self.get_ego_state_f()

        # Create planner
        self.planner = VehiclePlannerAcados2022Bilevel(
            ego_model_params=self.ego_model_params,
            road=self.current_road,
            planner_options=self.planner_options,
            opp_model_params=[],
            replan_threshold=np.array([10, 3, 1, 5, 10]),
        )

        self.planner.warm_start(states_ego=initial_state, actions=self.default_action)

        self.is_initialized = True

    def main_iteration(self):
        """Main planner iteration"""

        state = self.get_ego_state_f()

        self.planner.solve_bilevel(
            states_ego=state,
            actions=self.default_action,
            states_opp=[],
            current_time=self.time_now,
        )
        x_full_f = deepcopy(self.planner.x_full)
        u_full = deepcopy(self.planner.u_full)
        x_full_c = self.current_road.transform_trajectory_f2c(
            FrenetTrajectory(x_full_f)
        )
        # plt.plot(x_full_f[0, :], x_full_f[1, :])
        # plt.show()
        self.send_trajectory(x_full_c.get_as_array(), x_full_f, u_full)

    def odometry_callback(self, msg):
        self.odo_lock.acquire()
        self.odometry = deepcopy(msg)
        self.odo_lock.release()

    def path_callback(self, msg):
        self.path_lock.acquire()
        self.path = deepcopy(msg)
        self.path_lock.release()

    def position_on_track_callback(self, msg):
        self.position_on_track_lock.acquire()
        self.position_on_track = deepcopy(msg)
        self.position_on_track_lock.release()

    def object_list_callback(self, msg):
        self.object_lock.acquire()
        self.object_list = deepcopy(msg)
        self.object_lock.release()

    def vehicle_limits_callback(self, msg):
        self.vehicle_limits_lock.acquire()
        self.vehicle_limits = msg
        self.vehicle_limits_lock.release()


def main(args=None):
    print("Enter Main")
    rclpy.init(args=args)

    mpcrl = MPCRL()
    print("Spin Node")
    rclpy.spin(mpcrl)

    mpcrl.destroy_node()
    print("Shut down")
    rclpy.shutdown()


if __name__ == "__main__":
    main()
