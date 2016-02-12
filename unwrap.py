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
        raw_points = [[-0.31667 * self.width, +0.31094 * self.height],
                    [-0.01875 * self.width, +0.39219 * self.height],
                    [+0.27917 * self.width, +0.33125 * self.height],
                    [+0.29792 * self.width, -0.42188 * self.height],
                    [-0.03333 * self.width, -0.52969 * self.height],
                    [-0.28125 * self.width, -0.43594 * self.height]]

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
        # self.draw_ellipse(self.point_d, self.point_e, self.point_f)

    def draw_ellipse(self, left, top, right):
        aleft = np.array(left)
        atop = np.array(top)
        aright = np.array(right)

        # AVG between left and right points

        center_point = ((left[0] + right[0]) / 2, (left[1] + right[1]) / 2)
        acenter = np.array(center_point)

        axis = (int(np.linalg.norm(aleft - aright) / 2), int(np.linalg.norm(acenter - atop)))

        angle = np.arctan(aright - aleft)[0]

        print(aright - aleft, angle)
        cv2.ellipse(self.image, center_point, axis, -angle, 180, 360, color=self.COLOR)


if __name__ == '__main__':
    Main().run()