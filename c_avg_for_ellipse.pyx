#!python

cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.cdivision(True)
def get_avg_for_ellipse(np.ndarray[np.uint8_t, ndim=2] imcv, double a, double b, signed long sign,
                             signed long center_x, signed long center_y):
    cdef double val = 0
    cdef unsigned long width = int(a)
    cdef signed long x, y
    cdef double dx, temp
    for x from center_x - width <= x <= center_x + width:
        dx = center_x - x
        temp = (1 - dx * dx / (a * a)) ** 0.5
        y = int(sign * b * temp + center_y)
        val += (imcv[y, x] / (a * 2))
    return int(val)
