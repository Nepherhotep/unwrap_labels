import cv2
import numpy as np
from matplotlib import pyplot as plt


def round(value):
    return int(np.round(value))


class Line(object):
    def __init__(self, point1, point2):
        """
        For line formula y(x) = k * x + b, calc k and b params
        If the line is vertical, set "vertical" attr to True and save "x" position of the line
        """
        self.point1 = point1
        self.point2 = point2
        self.vertical = False
        self.fixed_x = None
        self.k = None
        self.b = None
        self.set_line_props(point1, point2)

    def is_vertical(self):
        return self.vertical

    def set_line_props(self, point1, point2):
        if point2[0] - point1[0]:
            self.k = float(point2[1] - point1[1]) / (point2[0] - point1[0])
            self.b = point2[1] - self.k * point2[0]
        else:
            self.vertical = True
            self.fixed_x = point2[0]

    def get_x(self, y):
        if self.is_vertical():
            return self.fixed_x
        else:
            return round(float(y - self.b) / self.k)

    def get_y(self, x):
        return self.k * x + self.b


class EdgeDetector(object):
    YELLOW_COLOR = (0, 255, 255)

    def __init__(self, src_image, pixel_points=[], percent_points=[]):
        """
        Points define two parallel lines - line A and line B
        TODO: make lines to be a separate class

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
        self.center_line = None

    def load_points(self):
        if not self.points:
            points = []
            for point in self.percent_points:
                x = round((point[0] + 0.5) * self.width)
                y = round((0.5 - point[1]) * self.height)
                points.append((x, y))

            self.points = np.array(points)
            print(self.points)

        if not len(self.points) == 4:
            raise ValueError('Array of points should have length == 4')

        self.point_a, self.point_b, self.point_c, self.point_d = self.points
        self.line_a = Line(self.point_a, self.point_d)
        self.line_b = Line(self.point_b, self.point_c)

    def calc_center_line(self):
        point1 = (self.point_a + self.point_b) / 2
        point2 = (self.point_d + self.point_c) / 2
        self.center_line = Line(point1, point2)

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
        self.calc_center_line()
        
        dst = self.apply_sobel_filter(self.src_image)
        matrix = self.build_matrix(dst)
        max_index = matrix.argmax()
        y_top, y_right = np.unravel_index(max_index, matrix.shape)
        cv2.imwrite('matrix.png', matrix)

        self.draw_mask(dst)
        top = (self.center_line.get_x(y_top), y_top)
        right = (self.line_b.get_x(y_right), y_right)

        cv2.imwrite('edges.jpg', dst)
        self.debug_point(self.src_image, top, self.YELLOW_COLOR)
        self.debug_point(self.src_image, right, self.YELLOW_COLOR)

        cv2.imwrite('out.jpg', self.src_image)

    def build_matrix(self, imcv):
        output = np.zeros((self.height, self.height))
        max_i, max_j, max_value = 0, 0, 0
        for i in range(self.height / 5):
            for j in range(self.height / 5):
                avg = self.get_avg_for_point(imcv, i, j)
                output[i, j] = avg
                if max_value < avg:
                    max_value = avg
                    max_i = i
                    max_j = j
        return output

    def get_avg_for_point(self, imcv, y_top, y_right):
        x_top = self.center_line.get_x(y_top)
        x_right = self.line_b.get_x(y_right)
        top = np.array([x_top, y_top])
        right = np.array([x_right, y_right])

        center = self.get_center_point(right)

        # get ellipse axis
        a = np.linalg.norm(center - right)
        b = np.linalg.norm(center - top)
        width = int(a)

        # get start and end angles
        if (top - center)[1] > 0:
            sign = 1

        else:
            sign = -1

        val = 0
        for x in range(round(center[0]) - width, round(center[0]) + width + 1):

            y = self.get_ellipse_y(a, b, center, sign, x)
            val += imcv[y, x] / float(width * 2)
        return val

    def get_ellipse_y(self, a, b, center, sign, x):
        dx = center[0] - x
        y = sign * b * (1 - dx * dx / (a * a)) ** 0.5 + center[1]
        return round(y)

    def get_ellipse_point(self, a, b, phi):
        """
        Get ellipse radius in polar coordinates
        """
        return a * np.cos(phi), b * np.sin(phi)

    def get_center_point(self, right):
        if self.center_line.is_vertical():
            x_center = self.center_line.get_x(right[1])
            y_center = self.center_line.get_y(x_center)
        else:
            k_center = self.center_line.k
            k_normal = - 1 / k_center
            b_normal = right[1] - k_normal * right[0]
            x_center = (b_normal - self.center_line.b) / (self.center_line.k - k_normal)
            y_center = k_normal * x_center + b_normal
        return int(x_center), int(y_center)

    def debug_point(self, imcv, point, color=255, thickness=3):
        cv2.line(imcv, tuple(point), tuple(point), color=color, thickness=thickness)

    def draw_mask(self, imcv):
        for point in self.points:
            self.debug_point(imcv, point)

        for line in [self.line_a, self.line_b, self.center_line]:
            get_point = lambda y: (line.get_x(y), y)
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
