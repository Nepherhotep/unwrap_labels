#!python

cimport numpy as np
cimport cython
from libc.math cimport sin, cos


cdef double PI = 3.14159265358979


@cython.boundscheck(False)
@cython.cdivision(True)
def get_avg_for_ellipse(np.ndarray[np.uint8_t, ndim=2] imcv, double a, double b, signed long sign,
                        signed long center_x, signed long center_y, double rot_cos, double rot_sin):
    # TODO: add ellipse angle
    cdef double val = 0
    cdef signed long x, y, points_count
    cdef double dx, dy, i

    points_count = int(a)
    cdef double delta = sign * PI / (points_count - 1)

    for i from 0 <= i < points_count:
        phi = i * delta
        dx = a * cos(phi)
        dy = b * sin(phi)

        x = int(dx * rot_cos - dy * rot_sin + center_x)
        y = int(dx * rot_sin + dy * rot_cos + center_y)

        val += (imcv[y, x] / points_count)

    return int(val)
