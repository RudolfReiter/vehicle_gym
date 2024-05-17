from typing import Tuple
import casadi as cs
import numpy as np


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


def halfspace(x1, y1, x2, y2):
    """
    computes a halfspace implicit equation of the form a*x + b*y + c >= 0 the halfplane to the "left" of two consequtive
    vectors (x1,y1) and (x2,y2).
    :param x1: coordinate of points on a line
    :param y1: coordinate of points on a line
    :param x2: coordinate of points on a line
    :param y2: coordinate of points on a line
    :return: parameters of line a*x + b*y + c >= 0
    """
    a = (y1 - y2)
    b = (x2 - x1)
    c = (x1 * y2 - x2 * y1)
    return a, b, c


def get_circle_rect_cover(n_circ: int, wheelbase: float, l_rect: float, w_rect: float, state_s, state_n, state_alpha):
    """
    Get n circles that cover vehicle
    :param n_circ:
    :param wheelbase:
    :param l_rect:
    :param w_rect:
    :param state_s:
    :param state_n:
    :param state_alpha:
    :return:
    """

    if type(state_s) == float:
        cosine = np.cos
        sine = np.sin
    else:
        cosine = cs.cos
        sine = cs.sin

    centers_s, centers_n = [], []
    radius = None
    d_rear = (l_rect - wheelbase) / 2
    if n_circ == 1:
        centers_s.append(state_s + wheelbase / 2 * cosine(state_alpha))
        centers_n.append(state_n + wheelbase / 2 * sine(state_alpha))
        radius = np.sqrt((l_rect / 2) ** 2 + (w_rect / 2) ** 2)
    if n_circ >= 2:
        d1 = w_rect / 2
        centers_s.append(state_s + (-d_rear + d1) * cosine(state_alpha))
        centers_n.append(state_n + (-d_rear + d1) * sine(state_alpha))

        centers_s.append(state_s + (d_rear + wheelbase - d1) * cosine(state_alpha))
        centers_n.append(state_n + (d_rear + wheelbase - d1) * sine(state_alpha))

        n_middle_circ = n_circ - 2
        for i in range(0, n_middle_circ):
            dist = (-d_rear + d1) + (i + 1) * (l_rect - w_rect) / (n_middle_circ + 1)
            centers_s.append(state_s + dist * cosine(state_alpha))
            centers_n.append(state_n + dist * sine(state_alpha))

        radius = np.sqrt((w_rect / 2) ** 2 + (w_rect / 2) ** 2)

    return tuple(centers_s), tuple(centers_n), radius

if __name__ == "__main__":
    p1 = (1,0)
    p2 = (1,-1)
    a,b,c = halfspace(p1[0], p1[1], p2[0],p2[1])
    test_point = (2,0)
    print(a*test_point[0]+b*test_point[1]+c)
