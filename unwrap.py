import sys
from pprint import pprint

import cv2
import numpy as np


class Main():
    COLOR = (255, 255, 255)

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
        self.point_a, self.point_b, self.point_c, self.point_d, self.point_e, self.point_f = points

    def save_image(self):
        cv2.imwrite('out.jpg', self.image)

    def run(self):
        self.load_image()
        self.load_points()
        self.draw_mask()
        self.save_image()

    def draw_poly_mask(self):
        cv2.polylines(self.image, np.int32([self.points]), 1, self.COLOR)

    def draw_mask(self):
        cv2.line(self.image, self.point_f, self.point_a, self.COLOR)
        cv2.line(self.image, self.point_c, self.point_d, self.COLOR)

        self.draw_ellipse(self.point_a, self.point_b, self.point_c)
        self.draw_ellipse(self.point_d, self.point_e, self.point_f)

    def draw_ellipse(self, left, top, right):
        """
        :type left: np.array
        :type top: np.array
        :type right: np.array
        """
        aleft = np.array(left)
        atop = np.array(top)
        aright = np.array(right)

        # AVG between left and right points
        acenter = (aleft + aright) / 2
        center_point = tuple(acenter.tolist())

        axis = (int(np.linalg.norm(aleft - aright) / 2), int(np.linalg.norm(acenter - atop)))

        x, y = aleft - aright
        angle = np.arctan(float(y) / x) * 57.3

        if (atop - acenter)[1] > 0:
            start_angle, end_angle = 0, 180
        else:
            start_angle, end_angle = 180, 360

        cv2.ellipse(self.image, center_point, axis, angle, start_angle, end_angle,
                    color=self.COLOR)


if __name__ == '__main__':
    Main().run()