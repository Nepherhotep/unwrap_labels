#!python

cimport numpy as np

def get_avg_for_ellipse(np.ndarray[np.uint8_t, ndim=2] imcv, float a, float b, int sign,
                             int center_x, int center_y):
    cdef float val = 0
    cdef int width = int(a)
    cdef int dx, y
    for x from int(center_x) - width <= x <= int(center_x) + width:
        dx = center_x - x
        y = int(sign * b * pow((1 - dx * dx / (a * a)), 0.5) + center_y)
        val += imcv[y, x] / float(width * 2)
    return int(val)
