import cv2
import numpy as np
from matplotlib import pyplot as plt


class EdgeDetector(object):
    def __init__(self, src_image, pixel_points=[], percent_points=[]):
        """
        Points define two parallel lines - line A and line B

        A         B
        |         |
        |         |
        |         |
        |         |
        D         C

        """
        self.src_image = src_image
        self.preprocessed_image = None
        self.width = self.src_image.shape[1]
        self.height = src_image.shape[0]
        self.points = pixel_points
        self.percent_points = percent_points

        self.point_a, self.point_b, self.point_c, self.point_d = (None, None, None, None)
        self.line_a, self.line_b = (None, None)

    def load_points(self):
        if not self.points:
            points = []
            for point in self.percent_points:
                x = self.round((point[0] + 0.5) * self.width)
                y = self.round((0.5 - point[1]) * self.height)
                points.append((x, y))

            self.points = np.array(points)

        if not len(self.points) == 4:
            raise ValueError('Array of points should have length == 4')

        self.point_a, self.point_b, self.point_c, self.point_d = self.points
        self.line_a = self.get_line_props(self.point_a, self.point_d)
        self.line_b = self.get_line_props(self.point_b, self.point_c)

    def get_line_props(self, point1, point2):
        """
        For line formula y(x) = k * x + b,
        return (k, b)
        """
        k = float(point2[1] - point1[1]) / (point2[0] - point1[0])
        b = point2[1] - k * point2[0]
        return (k, b)

    def apply_sobel_filter(self, imcv):
        """
        Apply sobel filter to view edges
        :param imcv: initial image
        :return: dst image
        """
        gray = cv2.cvtColor(imcv, cv2.COLOR_BGR2GRAY)
        scale = 1
        delta = 0
        ksize = 3
        ddepth = cv2.CV_16S

        # Gradient-X
        grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=ksize, scale=scale, delta=delta,
                           borderType=cv2.BORDER_DEFAULT)

        # Gradient-Y
        grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=ksize, scale=scale, delta=delta,
                           borderType=cv2.BORDER_DEFAULT)

        # converting back to uint8
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)

        # join to images into as single one
        return cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    def detect(self):
        self.load_points()
        dst = self.apply_sobel_filter(self.src_image)

        self.draw_mask(dst)
        cv2.imwrite('edges.jpg', dst)

    def round(self, value):
        return int(np.round(value))

    def draw_mask(self, imcv):
        for point in self.points:
            cv2.line(imcv, tuple(point), tuple(point), color=255, thickness=10)

        for line in [self.line_a, self.line_b]:
            get_point = lambda y: (self.round(float(y - line[1]) / line[0]), y)
            point1 = get_point(0)
            point2 = get_point(self.height)
            cv2.line(imcv, point1, point2, color=255, thickness=2)


if __name__ == '__main__':
    imcv = cv2.imread('image2.jpg', cv2.IMREAD_UNCHANGED)
    points = [[-0.45550, +0.20889],
              [+0.46992, +0.19538],
              [+0.46483, -0.16038],
              [-0.45507, -0.13624]]

    d = EdgeDetector(imcv, percent_points=points)
    d.detect()
