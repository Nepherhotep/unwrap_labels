import sys
from pprint import pprint

import cv2
import numpy as np


class Main():
    WHITE_COLOR = (255, 255, 255)
    YELLOW_COLOR = (0, 255, 255)

    def __init__(self):
        self.image = None
        self.width = None
        self.height = None
        self.points = None

    def load_image(self):
        self.image = cv2.imread('image.jpg', cv2.IMREAD_UNCHANGED)
        self.height, self.width, channels = self.image.shape

    def load_points(self):
        raw_points = [[-0.31667 * self.width, +0.31406 * self.height],
                      [-0.03750 * self.width, +0.39375 * self.height],
                      [+0.28125 * self.width, +0.31406 * self.height],
                      [+0.29792 * self.width, -0.41094 * self.height],
                      [-0.01250 * self.width, -0.51094 * self.height],
                      [-0.27917 * self.width, -0.42812 * self.height]]

        points = []
        for point in raw_points:
            x = int(point[0] + self.width / 2)
            y = int(self.height / 2 - point[1])
            points.append((x, y))

        self.points = np.array(points)
        (self.point_a, self.point_b, self.point_c,
         self.point_d, self.point_e, self.point_f) =self.points

    def save_image(self):
        cv2.imwrite('out.jpg', self.image)

    def run(self):
        self.load_image()
        self.load_points()
        self.draw_mask()
        self.calc_mask()
        self.save_image()

    def calc_mask(self):
        col_count = 30
        row_count = 20

        top_points = self.calc_ellipse_points(self.point_a, self.point_b, self.point_c, col_count)
        bottom_points = self.calc_ellipse_points(self.point_d, self.point_e, self.point_f, col_count)

        cols = []
        for i in range(col_count):
            top_point = top_points[i]
            bottom_point = bottom_points[i]

            delta = (top_point - bottom_point) / float(row_count - 1)

            col = []
            for j in range(row_count):
                point = top_point - delta * j
                col.append(point)

                x, y = map(int, point)
                cv2.line(self.image, (x, y), (x, y), color=self.YELLOW_COLOR, thickness=3)

            cols.append(col)

    def draw_poly_mask(self, color=WHITE_COLOR):
        cv2.polylines(self.image, np.int32([self.points]), 1, color)

    def draw_mask(self, color=WHITE_COLOR):
        cv2.line(self.image, tuple(self.point_f.tolist()), tuple(self.point_a.tolist()), color)
        cv2.line(self.image, tuple(self.point_c.tolist()), tuple(self.point_d.tolist()), color)

        self.draw_ellipse(self.point_a, self.point_b, self.point_c, color)
        self.draw_ellipse(self.point_d, self.point_e, self.point_f, color)

    def draw_ellipse(self, left, top, right, color=WHITE_COLOR):
        # AVG between left and right points
        center = (left + right) / 2
        center_point = tuple(center.tolist())

        axis = (int(np.linalg.norm(left - right) / 2), int(np.linalg.norm(center - top)))

        x, y = left - right
        angle = np.arctan(float(y) / x) * 57.3

        if (top - center)[1] > 0:
            start_angle, end_angle = 0, 180
        else:
            start_angle, end_angle = 180, 360

        cv2.ellipse(self.image, center_point, axis, angle, start_angle, end_angle, color=color)

    def calc_ellipse_points(self, left, top, right, points_count=30):
        center = (left + right) / 2

        # get ellipse axis
        a = np.linalg.norm(left - right) / 2
        b = np.linalg.norm(center - top)

        # get ellipse angle
        x, y = left - right
        angle = np.arctan(float(y) / x)

        # get start and end angles
        if (top - center)[1] > 0:
            delta = np.pi / (points_count - 1)

        else:
            delta = - np.pi / (points_count - 1)

        points = []
        for i in range(points_count):
            phi = i * delta
            dx, dy = self.get_ellipse_point(a, b, phi)

            x = int(dx + center[0])
            y = int(dy + center[1])

            cv2.line(self.image, (x, y), (x, y), color=self.YELLOW_COLOR, thickness=3)
            points.append([x, y])

        return np.array(points)

    def get_ellipse_point(self, a, b, phi):
        """
        Get ellipse radius in polar coordinates
        """
        return a * np.cos(phi), b * np.sin(phi)


if __name__ == '__main__':
    Main().run()