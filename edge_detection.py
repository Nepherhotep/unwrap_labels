import cv2
import cython
import numpy as np
import math
from matplotlib import pyplot as plt
import c_avg_for_ellipse
from trig_utils import Line
from unwrap import LabelUnwrapper


def round(value):
    return int(value + 0.5)


def get_avg_for_ellipse(imcv, a, b, sign, center_x, center_y):
    val = 0
    width = int(a)
    for x in xrange(round(center_x) - width, round(center_x) + width + 1):
        dx = center_x - x
        y = round(sign * b * (1 - dx * dx / (a * a)) ** 0.5 + center_y)
        val += imcv[y, x] / float(width * 2)
    return val


class EdgeDetector(object):
    YELLOW_COLOR = (0, 255, 255)
    WORKING_SIZE = 256

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
        self.src_image, self.scale_factor = self.resize_image(src_image)
        self.preprocessed_image = None
        self.width = self.src_image.shape[1]
        self.height = self.src_image.shape[0]
        self.points = pixel_points
        self.percent_points = percent_points

        self.point_a, self.point_b, self.point_c, self.point_d = (None, None, None, None)
        self.line_a, self.line_b = (None, None)
        self.center_line = None

    def resize_image(self, imcv):
        width = imcv.shape[1]
        height = imcv.shape[0]

        if width > height:
            new_width = self.WORKING_SIZE
            scale_factor = float(new_width) / width
            new_height = int(scale_factor * height)
        else:
            new_height = self.WORKING_SIZE
            scale_factor = float(new_height) / height
            new_width = int(scale_factor * width)

        return cv2.resize(imcv, (new_width, new_height)), scale_factor

    def load_points(self):
        if not self.points:
            points = []
            for point in self.percent_points:
                x = round((point[0] + 0.5) * self.width)
                y = round((0.5 - point[1]) * self.height)
                points.append((x, y))

            self.points = np.array(points)

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
        points = self.detect_points(dst, True)

        points2 = self.detect_points(dst, False)
        points2.reverse()

        points.extend(points2)
        return points

    def detect_points(self, imcv, top=True):
        matrix = self.build_matrix(imcv, top)
        max_index = matrix.argmax()

        if top:
            offset = 0
        else:
            offset = self.height / 2

        index = np.unravel_index(max_index, matrix.shape)
        y_top = index[0] + offset
        y_right = index[1] + offset
        cv2.imwrite('matrix-{}.png'.format('top' if top else 'bottom'), matrix)
        self.draw_mask(imcv)

        top = np.array((self.center_line.get_x(y_top), y_top))
        right = np.array((self.line_b.get_x(y_right), y_right))
        center_point = np.array(self.get_center_point(right))

        left = 2 * center_point - np.array(right)

        cv2.imwrite('edges.jpg', imcv)
        return map(np.int32, map(lambda x: x / self.scale_factor, [left, top, right]))

    def build_matrix(self, imcv, top=True):
        output = np.zeros((self.height / 2, self.height / 2))
        max_i, max_j, max_value = 0, 0, 0
        if top:
            from_height = 0
            to_height = self.height / 2
        else:
            from_height = self.height / 2
            to_height = self.height

        for i in range(from_height, to_height):
            for j in range(from_height, to_height):
                avg = self.get_avg_for_point(imcv, i, j)
                output[i - from_height, j - from_height] = avg
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

        # get start and end angles
        if (top - center)[1] > 0:
            sign = 1

        else:
            sign = -1

        #
        val = c_avg_for_ellipse.get_avg_for_ellipse(imcv, a, b, sign, center[0], center[1],
                                                    self.center_line.angle_cos,
                                                    self.center_line.angle_sin)
        return val

    def get_ellipse_point(self, a, b, phi):
        """
        Get ellipse radius in polar coordinates
        """
        return a * np.cos(phi), b * np.sin(phi)

    def get_center_point(self, right):
        if self.center_line.is_vertical():
            x_center = self.center_line.get_x(right[1])
            y_center = right[1]
        else:
            k_center = self.center_line.k
            k_normal = - 1 / k_center
            b_normal = right[1] - k_normal * right[0]
            x_center = (b_normal - self.center_line.b) / (self.center_line.k - k_normal)
            y_center = k_normal * x_center + b_normal
        return int(x_center), int(y_center)

    @staticmethod
    def debug_point(imcv, point, color=255, thickness=3):
        cv2.line(imcv, tuple(point), tuple(point), color=color, thickness=thickness)

    def draw_mask(self, imcv):
        for point in self.points:
            self.debug_point(imcv, point)

        for line in [self.line_a, self.line_b, self.center_line]:
            get_point = lambda y: (line.get_x(y), y)
            point1 = get_point(0)
            point2 = get_point(self.height)
            cv2.line(imcv, point1, point2, color=255, thickness=1)


if __name__ == '__main__':
    imcv = cv2.imread('image2.jpg', cv2.IMREAD_UNCHANGED)
    points = [[-0.45550, +0.20889],
              [+0.46992, +0.19538],
              [+0.46483, -0.16038],
              [-0.45507, -0.13624]]

    d = EdgeDetector(imcv, percent_points=points)
    result = d.detect()

    unwrapper = LabelUnwrapper(src_image=imcv, pixel_points=result)
    dst_image = unwrapper.unwrap()
    cv2.imwrite("dst-image.jpg", dst_image)

    for p in result:
        d.debug_point(imcv, p, d.YELLOW_COLOR, 10)
    cv2.imwrite('out.jpg', imcv)
