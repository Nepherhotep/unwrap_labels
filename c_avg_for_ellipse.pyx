#!python

cimport numpy as np
cimport cython
from libc.math cimport sin, cos


cdef double PI = 3.14159265358979

cdef unsigned int AVG_DIFF = 25
@cython.boundscheck(False)
@cython.cdivision(True)
def get_avg_for_ellipse(np.ndarray[np.uint8_t, ndim=2] imcv, double a, double b, signed long sign,
                        signed long center_x, signed long center_y, double rot_cos, double rot_sin):
    # TODO: add ellipse angle
    cdef double avg_val = 0
    cdef signed long x, y, points_count, val, dval
    cdef signed long prev_val = 0
    cdef double dx, dy, i

    points_count = int(a)
    cdef double delta = sign * PI / (points_count - 1)
    cdef double weight_factor = 1
    for i from 0 <= i < points_count:
        phi = i * delta
        dx = a * cos(phi)
        dy = b * sin(phi)

        x = int(dx * rot_cos - dy * rot_sin + center_x)
        y = int(dx * rot_sin + dy * rot_cos + center_y)

        val = imcv[y, x]

        if i > 0:
            dval = val - prev_val

            if dval > 0:
                weight_factor += dval
            else:
                weight_factor -= dval

        prev_val = val
        avg_val += (val / points_count)

    return int(avg_val * points_count * AVG_DIFF / weight_factor)
