#!/usr/bin/env python3
""" This file holds typical functions that are needed to transform ROS data

Autor: Rudolf Reiter
Date: Dez. 2020
"""
from dataclasses import dataclass
import rclpy
import sys
import numpy as np
from geometry_msgs.msg import Quaternion
from matplotlib import pyplot as plt
from arg_msgs.msg import ObjectList, Object, ObjectActionList, ObjectAction
from vehiclegym.automotive_datastructures import FrenetTrajectory, CartesianTrajectory


def quaternion2euler(quaternion):
    """
    Converts quaternion (w in last place) to euler roll, pitch, yaw
    quaternion = [x, y, z, w]
    Bellow should be replaced when porting for ROS 2 Python tf_conversions is done.
    """
    x = quaternion.x
    y = quaternion.y
    z = quaternion.z
    w = quaternion.w

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def find_nearest_idxs(array, value):
    """Find the two indexes, where the value is located in between"""
    array = np.asarray(array)
    diff = array - value
    abs_diff = np.abs(diff)
    idx = (abs_diff).argmin()
    if diff[idx] > 0:
        if idx > 0:
            idxs = [idx - 1, idx]
        else:
            idxs = [idx, idx + 1]
    else:
        if idx < len(array) - 1:
            idxs = [idx, idx + 1]
        else:
            idxs = [idx - 1, idx]
    return idxs


def is_object_in_range(waypoints, miphs_object, margin_begin=0, margin_end=0):
    """Check if object is in waypoint range"""
    idxs = []
    for point_i in range(len(miphs_object.p)):
        idxs = idxs + find_nearest_idxs(
            waypoints.s, miphs_object.p[point_i].s
        )  # append
    if (
        int(0) in idxs
        or len(waypoints.s) - 1 in idxs
        or (waypoints.s[min(idxs)] - waypoints.s[0]) < margin_begin
    ):
        return False
    else:
        return True


def euler_from_quaternion(orientation):
    """Transforms quaternions to euler coordinates

    Args:
        orientation (array[4]): quaternions

    Returns:
        e: euler coordinates"""

    q = Quaternion()
    q.w = orientation[3]
    q.x = orientation[0]
    q.y = orientation[1]
    q.z = orientation[2]
    e = quaternion2euler(q)
    return e


def get_Orientation(pose):
    """Transforms pose type, which is used in the ROS framework to euler angles

    Args:
        pose: pose must have a field orientation, which holds quaterions

    Returns:
        euler angles"""
    orientation_q = pose.orientation
    orientation_list = [
        orientation_q.x,
        orientation_q.y,
        orientation_q.z,
        orientation_q.w,
    ]
    (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
    return (roll, pitch, yaw)


def get_ros_data_vehicle_state(odometry):
    """Get vehicle states out of ros datatype

    Args:
        orientation as represented in ros

    Returns:
        vehicle states
    """
    devbot_x = odometry.pose.pose.position.x
    devbot_y = odometry.pose.pose.position.y
    devbot_z = odometry.pose.pose.position.z
    (roll, pitch, yaw) = get_Orientation(odometry.pose.pose)
    devbot_psi = yaw
    devbot_v = odometry.twist.twist.linear.x
    devbot_yaw_rate = odometry.twist.twist.angular.z
    return devbot_x, devbot_y, devbot_z, devbot_psi, devbot_v, devbot_yaw_rate


def get_ros_waypoints(ros_path):
    """Transforms ros path data into waypoint struct and subtracts distance2borders safety margin

    Args:
        ros_path: ros path data type
        distance2borders: distance to borders which will be subtracted. e.g. the vehicle width or a safety distance

    Returns:
        way_point_data: object, that contains the waypoint data
        tracklen: length of the track
        overlap_action: indicates, wether some path values s were increased by the lap length, due overlap
    """
    track_s = np.array([point.s for point in ros_path.points])
    track_psi = np.array([get_Orientation(point.pose)[2] for point in ros_path.points])
    track_x = np.array([point.pose.position.x for point in ros_path.points])
    track_y = np.array([point.pose.position.y for point in ros_path.points])
    track_dl = np.array([point.dl for point in ros_path.points])
    track_dr = np.array([point.dr for point in ros_path.points])
    track_kappa = np.array([point.kappa for point in ros_path.points])
    track_vmax = np.array([point.v_max for point in ros_path.points])

    tracklen = max(track_s) + (track_s[-1] - track_s[-2])

    if track_s[0] > track_s[-1]:  # leads over start :(
        idx_start = np.argwhere(np.diff(track_s) < 0)[0][0] + 1
        overlap_action = True
        track_s[idx_start:] = track_s[idx_start:] + tracklen
        if len(np.argwhere(np.diff(track_s) < 0)) != 0:
            print("In function: get_ros_waypoints, ros_conversion_tools")
            print("Somewhere the track difference is bigger than 1: ", track_s)
    else:
        overlap_action = False

    track_x = np.expand_dims(track_x, 1)
    track_y = np.expand_dims(track_y, 1)
    track_xy = np.hstack((track_x, track_y))

    track_data = [
        track_xy,
        track_psi,
        track_s,
        track_kappa,
        track_dl,
        track_dr,
        track_vmax,
    ]

    return track_data, tracklen, overlap_action


def miphs_objects2ros_actions(miphs_objects, obst_dir, orig_ros_object_list, time_now):
    """Transforms miphs objects and its decisions into an ros action list
    #TODO: the obstacle directions should be integrated into the miphs object

        Args:
            miphs_objects: miphs object list
            obst_dir: boleans alighen with miphs objects, that indicat the action left right or catch ignore
            orig_ros_object_list: original ros objects to get some data from

        Returns:
            object_action_list: ros roborace type message, that represents the object plus an action
    """
    object_action_list = ObjectActionList()
    object_action_list.header = orig_ros_object_list.header
    object_action_list.header.stamp = time_now

    for i in np.arange(len(miphs_objects)):
        object_action = ObjectAction()
        object_action.header = object_action_list.header
        id_array = [object.obj_nr for object in miphs_objects]
        idx = [
            k for k in range(len(id_array)) if id_array[k] == miphs_objects[i].obj_nr
        ]
        object_action.object = orig_ros_object_list.objects[idx[0]]
        if miphs_objects[i].type == 0 and obst_dir[i] == 0:
            object_action.action = ObjectAction.action_LEFT
        elif miphs_objects[i].type == 0 and obst_dir[i] == 1:
            object_action.action = ObjectAction.action_RIGHT
        elif miphs_objects[i].type == 1 and obst_dir[i] == 1:
            object_action.action = ObjectAction.action_COLLECT
        else:
            object_action.action = ObjectAction.action_IGNORE
        object_action_list.objects_action.append(object_action)
    return object_action_list


def get_relevant_bounds(
    relevant_bound_canditate, n_same_side, n_opposite_side, min_dist
):
    """If a bound should be changed, it needs to be checked with the opposite bound not block the road and with the
    same side, to not make the road bigger"""
    return max(min(n_same_side, relevant_bound_canditate), -n_opposite_side + min_dist)


def miphs_objects2boundaries(
    miphs_objects,
    obst_dir,
    waypoints,
    add_dist_obstacle=0,
    add_dist_bonus=0,
    show_plot=False,
    min_dist_ignore_obstacles=0,
):
    """Transforms miphs objects and its decisions inside waypoint boundaries
    #TODO: the obstacle directions should be integrated into the miphs object

        Args:
            miphs_objects: miphs object list
            obst_dir: booleans aligned with miphs objects, that indicate the action left right or catch ignore
            waypoints: waypoints, where boundaries are transformed by reference(!)
            add_dist_obstacle: (optional) adds a distance (e.g. safety distance) to obstacle borders
            add_dist_bonus: (optional) adds a distance to bonuses. (e.g. could be negative to be sure)
            show_plot: plots the transformation for debugging issues
    """
    any_obj_in_range = False
    min_dist = 0.1
    if len(obst_dir) == 0:
        return

    for i in np.arange(len(miphs_objects)):
        # skip loop, if any point of the obstacle is outside
        # TODO: check if that works for all scenarios?

        if is_object_in_range(waypoints, miphs_objects[i], min_dist_ignore_obstacles):
            any_obj_in_range = True
            # if obstacles types
            if miphs_objects[i].type == 0:
                # if action left
                if obst_dir[i] == 0:
                    for point_i in range(len(miphs_objects[i].p)):
                        idxs = find_nearest_idxs(
                            waypoints.s, miphs_objects[i].p[point_i].s
                        )
                        relevant_obstacle_bound_candidte = (
                            -miphs_objects[i].p[point_i].n - add_dist_obstacle
                        )
                        waypoints.nr[idxs[0]] = get_relevant_bounds(
                            relevant_obstacle_bound_candidte,
                            waypoints.nr[idxs[0]],
                            waypoints.nl[idxs[0]],
                            min_dist,
                        )

                        waypoints.nr[idxs[1]] = get_relevant_bounds(
                            relevant_obstacle_bound_candidte,
                            waypoints.nr[idxs[1]],
                            waypoints.nl[idxs[1]],
                            min_dist,
                        )

                # if action right
                else:
                    for point_i in range(len(miphs_objects[i].p)):
                        idxs = find_nearest_idxs(
                            waypoints.s, miphs_objects[i].p[point_i].s
                        )
                        relevant_obstacle_bound_candidte = (
                            miphs_objects[i].p[point_i].n - add_dist_obstacle
                        )
                        waypoints.nl[idxs[0]] = get_relevant_bounds(
                            relevant_obstacle_bound_candidte,
                            waypoints.nl[idxs[0]],
                            waypoints.nr[idxs[0]],
                            min_dist,
                        )
                        waypoints.nl[idxs[1]] = get_relevant_bounds(
                            relevant_obstacle_bound_candidte,
                            waypoints.nl[idxs[1]],
                            waypoints.nr[idxs[1]],
                            min_dist,
                        )

            # if bonus type
            else:
                # if action collect
                if obst_dir[i] == 1:
                    idxs = find_nearest_idxs(waypoints.s, miphs_objects[i].poupp.s)
                    relevant_bonus_upper_bound = (
                        miphs_objects[i].poupp.n + add_dist_bonus
                    )
                    waypoints.nl[idxs[0]] = get_relevant_bounds(
                        relevant_bonus_upper_bound,
                        waypoints.nl[idxs[0]],
                        waypoints.nr[idxs[0]],
                        min_dist,
                    )
                    waypoints.nl[idxs[1]] = get_relevant_bounds(
                        relevant_bonus_upper_bound,
                        waypoints.nl[idxs[1]],
                        waypoints.nr[idxs[1]],
                        min_dist,
                    )

                    idxs = find_nearest_idxs(waypoints.s, miphs_objects[i].polow.s)
                    relevant_bonus_lower_bound = (
                        -miphs_objects[i].polow.n + add_dist_bonus
                    )
                    waypoints.nr[idxs[0]] = get_relevant_bounds(
                        relevant_bonus_lower_bound,
                        waypoints.nr[idxs[0]],
                        waypoints.nl[idxs[0]],
                        min_dist,
                    )
                    waypoints.nr[idxs[1]] = get_relevant_bounds(
                        relevant_bonus_lower_bound,
                        waypoints.nr[idxs[1]],
                        waypoints.nl[idxs[1]],
                        min_dist,
                    )

                # if action is "ignore" do nothing

    if show_plot and any_obj_in_range:
        N_obj = len(miphs_objects)
        objects = miphs_objects
        plt.plot(waypoints.s, waypoints.nl, "grey", label="bounds")
        plt.plot(waypoints.s, -waypoints.nr, "grey")

        plotted_obst = 0
        plotted_bon = 0
        for i in np.arange(N_obj):
            if objects[i].type == 0:
                if plotted_obst == 0:
                    plt.plot(
                        [objects[i].p[0].s, objects[i].p[1].s],
                        [objects[i].p[0].n, objects[i].p[1].n],
                        "r",
                        linewidth=1,
                        label="obstacle",
                    )
                    plotted_obst = 1
                plt.plot(
                    [objects[i].p[0].s, objects[i].p[1].s],
                    [objects[i].p[0].n, objects[i].p[1].n],
                    "r",
                    linewidth=1,
                )
                plt.plot(
                    [objects[i].p[1].s, objects[i].p[2].s],
                    [objects[i].p[1].n, objects[i].p[2].n],
                    "r",
                    linewidth=1,
                )
                plt.plot(
                    [objects[i].p[2].s, objects[i].p[3].s],
                    [objects[i].p[2].n, objects[i].p[3].n],
                    "r",
                    linewidth=1,
                )
                plt.plot(
                    [objects[i].p[3].s, objects[i].p[0].s],
                    [objects[i].p[3].n, objects[i].p[0].n],
                    "r",
                    linewidth=1,
                )
            else:
                if plotted_bon == 0:
                    plt.plot(
                        [objects[i].p[0].s, objects[i].p[1].s],
                        [objects[i].p[0].n, objects[i].p[1].n],
                        "g",
                        linewidth=1,
                        label="bonus",
                    )
                    plotted_bon = 1
                plt.plot(
                    [objects[i].p[0].s, objects[i].p[1].s],
                    [objects[i].p[0].n, objects[i].p[1].n],
                    "g",
                    linewidth=1,
                )
                plt.plot(
                    [objects[i].p[1].s, objects[i].p[2].s],
                    [objects[i].p[1].n, objects[i].p[2].n],
                    "g",
                    linewidth=1,
                )
                plt.plot(
                    [objects[i].p[2].s, objects[i].p[3].s],
                    [objects[i].p[2].n, objects[i].p[3].n],
                    "g",
                    linewidth=1,
                )
                plt.plot(
                    [objects[i].p[3].s, objects[i].p[0].s],
                    [objects[i].p[3].n, objects[i].p[0].n],
                    "g",
                    linewidth=1,
                )

        plt.plot()
        plt.legend()
        plt.grid()
        plt.show()


def downsample_waypoints(way_point_data, idx_max, n_downsample):
    """Downsample waypoints on a s_grid (1m sample distance)"""
    s0 = round(way_point_data.s[0])
    mod_s0 = np.mod(s0, n_downsample)
    s_add = n_downsample - mod_s0

    s_add = int(s_add)
    idx_max = int(idx_max)
    n_downsample = int(n_downsample)

    way_point_data.s = way_point_data.s[s_add : s_add + idx_max : n_downsample]
    way_point_data.p = way_point_data.p[s_add : s_add + idx_max : n_downsample, :]
    way_point_data.nl = way_point_data.nl[s_add : s_add + idx_max : n_downsample]
    way_point_data.nr = way_point_data.nr[s_add : s_add + idx_max : n_downsample]
    way_point_data.kappa = way_point_data.kappa[s_add:idx_max:n_downsample]
    way_point_data.psi = way_point_data.psi[s_add : s_add + idx_max : n_downsample]


def update_waypoints_with_old_bounds(
    way_point_data, old_way_point_data, s_fix_horizon, increase_boundary
):
    if old_way_point_data is None:
        return
    for idx in range(len(old_way_point_data.s)):
        if old_way_point_data.s[idx] > way_point_data.s[0] + s_fix_horizon:
            return
        way_point_data.nl[
            np.abs(way_point_data.s - old_way_point_data.s[idx]) < 0.001
        ] = (old_way_point_data.nl[idx] + increase_boundary)
        way_point_data.nr[
            np.abs(way_point_data.s - old_way_point_data.s[idx]) < 0.001
        ] = (old_way_point_data.nr[idx] + increase_boundary)
