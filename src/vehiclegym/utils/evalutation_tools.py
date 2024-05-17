from typing import Tuple, List
import numpy as np
from vehicle_models.model_kinematic import KinematicModelParameters
from vehiclegym.utils.automotive_datastructures import FrenetTrajectory
from vehiclegym.road.road import Road, RoadOptions


def is_on_right_side(x, y, xy0, xy1):
    x0, y0 = xy0
    x1, y1 = xy1
    a = float(y1 - y0)
    b = float(x0 - x1)
    c = - a * x0 - b * y0
    return a * x + b * y + c >= 0


def test_point(x, y, vertices):
    num_vert = len(vertices)
    is_right = [is_on_right_side(x, y, vertices[i], vertices[(i + 1) % num_vert]) for i in range(num_vert)]
    all_left = not any(is_right)
    all_right = all(is_right)
    return all_left or all_right


def get_vertices_rectangle(pos_center: Tuple[float, float], dims: Tuple[float, float, float]):
    (length, width, angle) = dims
    (pos_x, pos_y) = pos_center

    # 2x1 matrix
    offset = np.array([[pos_x], [pos_y]])

    # 2x4 matrix
    vertices = np.transpose(np.array([[length / 2, width / 2],
                                      [-length / 2, width / 2],
                                      [-length / 2, -width / 2],
                                      [length / 2, -width / 2]]))

    # 2x2 rotation matrix
    rotation_mat = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

    return rotation_mat @ vertices + offset


def check_collision_cartesian(pos_center_veh_1: Tuple[float, float],
                              dims_veh_1: Tuple[float, float, float],
                              pos_center_veh_2: Tuple[float, float],
                              dims_veh_2: Tuple[float, float, float]) -> bool:
    """
    Checks for a collision of two vehicles
    :param pos_center_veh_1: tuple with center position (x,y) of vehicle
    :param dims_veh_1: dimension tuple (length, width, angle) of vehicle
    :param pos_center_veh_2: tuple with center position (x,y) of vehicle
    :param dims_veh_2: dimension tuple (length, width, angle) of vehicle
    :return: boolean variable indicates if collision occurred
    """

    vertices_veh_1 = get_vertices_rectangle(pos_center_veh_1, dims_veh_1)
    vertices_veh_2 = get_vertices_rectangle(pos_center_veh_2, dims_veh_2)

    vertices_veh_1_tuples = [(vertices_veh_1[0, i], vertices_veh_1[1, i]) for i in range(4)]
    vertices_veh_2_tuples = [(vertices_veh_2[0, i], vertices_veh_2[1, i]) for i in range(4)]

    collision = False
    for i in range(4):
        point_x = vertices_veh_1[0, i]
        point_y = vertices_veh_1[1, i]
        collision = collision or test_point(x=point_x, y=point_y, vertices=vertices_veh_2_tuples)

    for i in range(4):
        point_x = vertices_veh_2[0, i]
        point_y = vertices_veh_2[1, i]
        collision = collision or test_point(x=point_x, y=point_y, vertices=vertices_veh_1_tuples)

    return collision


def check_collision_cartesian_ellipsoid(
        pos_center_veh_1: Tuple[float, float],
        dims_veh_1: Tuple[float, float, float],
        pos_center_veh_2: Tuple[float, float],
        dims_veh_2: Tuple[float, float, float],
        ego_parameter: KinematicModelParameters = None) -> bool:
    """
    Checks for a collision of two vehicles, where one vehilce is represented
    as circle and the other as ellipsis
    :param pos_center_veh_1: tuple with center position (x,y) of vehicle
    :param dims_veh_1: dimension tuple (length, width, angle) of vehicle
    :param pos_center_veh_2: tuple with center position (x,y) of vehicle
    :param dims_veh_2: dimension tuple (length, width, angle) of vehicle
    :return: boolean variable indicates if collision occurred
    """
    # use not actual radius but distances defined in ego vehilce parameters
    # r_veh_1 = np.sqrt((dims_veh_1[0] / 2) ** 2 + (dims_veh_1[1] / 2) ** 2)
    # a = (dims_veh_2[0] / 2. + r_veh_1) * np.power(2, 1 / 2)
    # b = (dims_veh_2[1] / 2. + r_veh_1) * np.power(2, 1 / 2)
    if ego_parameter is None:
        r_veh_1 = np.sqrt((dims_veh_1[0] / 2) ** 2 + (dims_veh_1[1] / 2) ** 2)
        a = dims_veh_2[0] / np.sqrt(2) + r_veh_1
        b = dims_veh_2[1] / np.sqrt(2) + r_veh_1
    else:
        a = dims_veh_2[0] / np.sqrt(2) + ego_parameter.safety_radius_length*0.99
        b = dims_veh_2[1] / np.sqrt(2) + ego_parameter.safety_radius_side*0.99

    # main axis matrix
    D = np.array([[1 / (a ** 2), 0],
                  [0, 1 / (b ** 2)]])

    # rotation matrix
    R = np.array([[np.cos(dims_veh_2[2]), -np.sin(dims_veh_2[2])],
                  [np.sin(dims_veh_2[2]), np.cos(dims_veh_2[2])]])

    # center matrix
    K = np.array([[pos_center_veh_2[0]],
                  [pos_center_veh_2[1]]])

    # compact state
    center_ego = np.array([[pos_center_veh_1[0]],
                           [pos_center_veh_1[1]]])

    condition = np.sqrt(np.transpose(center_ego - K) @ R @ D @ np.transpose(R) @ (center_ego - K)) - 1
    collision = condition[0][0]<0
    #norm_vec = np.transpose(R) @ (center_ego - K)
    #norm_val = (np.fabs(norm_vec[0]) / a) ** 2 + (np.fabs(norm_vec[1]) / b) ** 2
    #norm_val = np.power(norm_val, 1 / 2)
    #collision = (norm_val[0] - 1) < 0

    return collision


def get_minimum_distance(pos_center_veh_1: Tuple[float, float],
                         dims_veh_1: Tuple[float, float, float],
                         pos_center_veh_2: Tuple[float, float],
                         dims_veh_2: Tuple[float, float, float]) -> float:
    vertices_veh_1 = get_vertices_rectangle(pos_center_veh_1, dims_veh_1)
    vertices_veh_2 = get_vertices_rectangle(pos_center_veh_2, dims_veh_2)

    total_min = 1e6
    for i in range(4):
        vertices_veh_2_rolled = np.roll(vertices_veh_2, i, axis=1)
        diff_sqr = (vertices_veh_1 - vertices_veh_2_rolled) ** 2
        dist = np.sqrt(diff_sqr[0, :] + diff_sqr[1, :])
        min_dist = np.min(dist)
        total_min = np.minimum(min_dist, total_min)
    return total_min


def check_collision_frenet(state_veh_1: Tuple[float, float, float],
                           dims_veh_1: Tuple[float, float],
                           state_veh_2: Tuple[float, float, float],
                           dims_veh_2: Tuple[float, float],
                           road: Road,
                           center_offset_veh_1: float = 0.,
                           center_offset_veh_2: float = 0.,
                           check_ellipsoidal: bool = False,
                           ego_parameter: KinematicModelParameters = None) -> bool:
    frenet_state_1 = FrenetTrajectory(np.array([*state_veh_1, 0, 0]))
    frenet_state_2 = FrenetTrajectory(np.array([*state_veh_2, 0, 0]))

    c_state_1 = road.transform_trajectory_f2c(frenet_state_1)
    c_state_2 = road.transform_trajectory_f2c(frenet_state_2)

    pos_center_veh_1 = (c_state_1.x[0] - center_offset_veh_1 * np.cos(c_state_1.phi[0]),
                        c_state_1.y[0] - center_offset_veh_1 * np.sin(c_state_1.phi[0]))

    pos_center_veh_2 = (c_state_2.x[0] - center_offset_veh_2 * np.cos(c_state_2.phi[0]),
                        c_state_2.y[0] - center_offset_veh_2 * np.sin(c_state_2.phi[0]))

    dim_veh_1 = (*dims_veh_1, c_state_1.phi[0])
    dim_veh_2 = (*dims_veh_2, c_state_2.phi[0])

    collision = (
        check_collision_cartesian_ellipsoid(pos_center_veh_1=pos_center_veh_1,
                                            dims_veh_1=dim_veh_1,
                                            pos_center_veh_2=pos_center_veh_2,
                                            dims_veh_2=dim_veh_2,
                                            ego_parameter=ego_parameter)
        if check_ellipsoidal else
        check_collision_cartesian(pos_center_veh_1=pos_center_veh_1,
                                  dims_veh_1=dim_veh_1,
                                  pos_center_veh_2=pos_center_veh_2,
                                  dims_veh_2=dim_veh_2)
    )

    return collision


def get_minimum_distance_frenet(state_veh_1: Tuple[float, float, float],
                                dims_veh_1: Tuple[float, float],
                                state_veh_2: Tuple[float, float, float],
                                dims_veh_2: Tuple[float, float],
                                road: Road,
                                center_offset_veh_1: float = 0.,
                                center_offset_veh_2: float = 0.) -> float:
    frenet_state_1 = FrenetTrajectory(np.array([*state_veh_1, 0, 0]))
    frenet_state_2 = FrenetTrajectory(np.array([*state_veh_2, 0, 0]))

    c_state_1 = road.transform_trajectory_f2c(frenet_state_1)
    c_state_2 = road.transform_trajectory_f2c(frenet_state_2)

    pos_center_veh_1 = (c_state_1.x[0] - center_offset_veh_1 * np.cos(c_state_1.phi[0]),
                        c_state_1.y[0] - center_offset_veh_1 * np.sin(c_state_1.phi[0]))

    pos_center_veh_2 = (c_state_2.x[0] - center_offset_veh_2 * np.cos(c_state_2.phi[0]),
                        c_state_2.y[0] - center_offset_veh_2 * np.sin(c_state_2.phi[0]))

    dim_veh_1 = (*dims_veh_1, c_state_1.phi[0])
    dim_veh_2 = (*dims_veh_2, c_state_2.phi[0])

    return get_minimum_distance(pos_center_veh_1=pos_center_veh_1,
                                dims_veh_1=dim_veh_1,
                                pos_center_veh_2=pos_center_veh_2,
                                dims_veh_2=dim_veh_2)


def get_min_distance(ego_states: np.ndarray, static_obstacles: List[np.ndarray]):
    min_dist = 1e6
    for static_obstacle in static_obstacles:
        dists = np.sqrt(((ego_states[:2,:]-np.expand_dims(static_obstacle[0:2],axis=1))**2).sum(axis=0))
        min_dist = np.minimum(np.min(dists), min_dist)
    return min_dist

def check_collision_multi(ego_state: np.ndarray,
                          ego_parameter: KinematicModelParameters,
                          opp_states: List[np.ndarray],
                          opp_parameters: List[KinematicModelParameters],
                          road: Road,
                          check_ellipsoidal: bool = False) -> [bool, List[int], bool]:
    collision = False
    state_ego = tuple(ego_state[0:3])
    dim_ego = (ego_parameter.chassis_length, ego_parameter.chassis_width)
    offset_ego = -ego_parameter.length / 2
    collision_idxs = []
    is_ego_lead = None
    for i, [opp_state, opp_parameter] in enumerate(zip(opp_states, opp_parameters)):
        state_opp = tuple(opp_state[0:3])
        dim_opp = (opp_parameter.chassis_length, opp_parameter.chassis_width)
        offset_opp = -opp_parameter.length / 2
        current_collision = check_collision_frenet(state_veh_1=state_ego,
                                                   dims_veh_1=dim_ego,
                                                   state_veh_2=state_opp,
                                                   dims_veh_2=dim_opp,
                                                   road=road,
                                                   center_offset_veh_1=offset_ego,
                                                   center_offset_veh_2=offset_opp,
                                                   check_ellipsoidal=check_ellipsoidal,
                                                   ego_parameter=ego_parameter)
        if current_collision:
            collision_idxs.append(i)
            if ego_state[0] > opp_state[0]:
                is_ego_lead = True
            else:
                is_ego_lead = False

        collision = collision or current_collision
    return collision, collision_idxs, is_ego_lead


def get_minimum_distance_multi(ego_state: np.ndarray,
                               ego_parameter: KinematicModelParameters,
                               opp_states: List[np.ndarray],
                               opp_parameters: List[KinematicModelParameters],
                               road: Road) -> [float, List[int]]:
    distance = 1e6
    state_ego = tuple(ego_state[0:3])
    dim_ego = (ego_parameter.chassis_length, ego_parameter.chassis_width)
    offset_ego = -ego_parameter.length / 2
    min_distance_idx = -1

    for i, [opp_state, opp_parameter] in enumerate(zip(opp_states, opp_parameters)):
        state_opp = tuple(opp_state[0:3])
        dim_opp = (opp_parameter.chassis_length, opp_parameter.chassis_width)
        offset_opp = -opp_parameter.length / 2
        current_distance = get_minimum_distance_frenet(state_veh_1=state_ego,
                                                       dims_veh_1=dim_ego,
                                                       state_veh_2=state_opp,
                                                       dims_veh_2=dim_opp,
                                                       road=road,
                                                       center_offset_veh_1=offset_ego,
                                                       center_offset_veh_2=offset_opp)
        if current_distance < distance:
            min_distance_idx = i
            distance = current_distance

    return distance, min_distance_idx


if __name__ == "__main__":
    is_collision_rect = check_collision_cartesian((0, 0), (10, 10, 0), (0, 7), (2, 2, 0))
    is_collision_ell = check_collision_cartesian_ellipsoid((0, 0), (10, 10, 0), (0, 7), (2, 2, 0))
    print(f"Collision_rect: {is_collision_rect}, collision ell: {is_collision_ell}")

    road_options_local = RoadOptions()
    road = Road(road_options_local)

    pars = KinematicModelParameters()
    pars.chassis_length = 4
    pars.chassis_width = 2
    pars.length_rear = 1
    pars.length_front = 1
    crash, opp_list, is_ego_lead = check_collision_multi(np.array([40 + 5.0, 0, 0, 0, 0]),
                                                         ego_parameter=pars,
                                                         opp_states=[np.array([40 + .2, 0, 0, 0, 0]),
                                                                     np.array([10, 5, 0, 0, 0])],
                                                         opp_parameters=[pars, pars],
                                                         road=road)
    print("Crash: {}, opp_list: {}, is_ego_lead: {}".format(crash, opp_list, is_ego_lead))

    are_collisions = check_collision_frenet(state_veh_1=(1, 0, 0),
                                            dims_veh_1=(2, 1),
                                            state_veh_2=(3.1, 0, 0),
                                            dims_veh_2=(2, 1),
                                            road=road,
                                            center_offset_veh_1=-1)

    dist = get_minimum_distance_multi(ego_state=np.array([1, 0, 0, 0, 0]),
                                      ego_parameter=KinematicModelParameters(),
                                      opp_states=[np.array([10, 0, 0, 0, 0]), np.array([0, 5, 0, 0, 0])],
                                      opp_parameters=[KinematicModelParameters(), KinematicModelParameters()],
                                      road=road)
    print(dist)
