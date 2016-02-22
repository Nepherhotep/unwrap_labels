#!python

cimport numpy as np
cimport cython
from libc.math cimport sin, cos


cdef double PI = 3.14159265358979


@cython.boundscheck(False)
@cython.cdivision(True)
def get_avg_for_ellipse(np.ndarray[np.uint8_t, ndim=2] imcv, double a, double b, signed long sign,
                        double angle, signed long center_x, signed long center_y,
                        unsigned int debug):
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

        x = int(dx * cos(angle) - dy * sin(angle) + center_x)
        y = int(dx * sin(angle) + dy * cos(angle) + center_y)

        val += (imcv[y, x] / (a * 2))

        if debug == 1:
            imcv[y][x] = 255


    return int(val)
