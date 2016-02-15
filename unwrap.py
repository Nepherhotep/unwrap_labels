import sys
from pprint import pprint

import cv2
import numpy as np

try:
    from scipy.interpolate import griddata
except ImportError:
    pass


class Main():
    COL_COUNT = 30
    ROW_COUNT = 20

    WHITE_COLOR = (255, 255, 255)
    YELLOW_COLOR = (0, 255, 255)

    def __init__(self):
        self.src_image = None
        self.dst_image = None
        self.width = None
        self.height = None
        self.points = None

        self.point_a = None  # top left
        self.point_b = None  # top center
        self.point_c = None  # top right
        self.point_d = None  # bottom right
        self.point_e = None  # bottom center
        self.point_f = None  # bottom left

    def load_image(self):
        self.src_image = cv2.imread('image.jpg', cv2.IMREAD_UNCHANGED)
        self.height, self.width, channels = self.src_image.shape

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
         self.point_d, self.point_e, self.point_f) = self.points

    def save_image(self):
        cv2.imwrite('out.jpg', self.src_image)
        cv2.imwrite('dst-perspective.jpg', self.dst_image)

    def run(self):
        self.load_image()
        self.load_points()


        source_map = self.calc_source_map()
        self.unwrap_label_perspective(source_map)
        self.unwrap_label_interpolation(source_map)
        self.draw_mask()
        self.save_image()

    def calc_dest_map(self):
        width, height = self.get_label_size()

        dx = float(width) / (self.COL_COUNT - 1)
        dy = float(height) / (self.ROW_COUNT - 1)

        rows = []
        for row_index in range(self.ROW_COUNT):
            row = []
            for col_index in range(self.COL_COUNT):
                row.append([int(dx * col_index),
                            int(dy * row_index)])

            rows.append(row)
        return np.array(rows)

    def unwrap_label_interpolation(self, source_map):
        """
        Unwrap label using interpolation - more accurate method in terms of quality
        """
        width, height = self.get_label_size()

        dest_map = self.calc_dest_map()

        grid_x, grid_y = np.mgrid[0:width - 1:width * 1j, 0:height - 1:height * 1j]

        destination = dest_map.reshape(dest_map.size / 2, 2)
        source = source_map.reshape(source_map.size / 2, 2)

        grid_z = griddata(destination, source, (grid_x, grid_y), method='cubic')
        map_x = np.append([], [ar[:, 0] for ar in grid_z]).reshape(width, height)
        map_y = np.append([], [ar[:, 1] for ar in grid_z]).reshape(width, height)
        map_x_32 = map_x.astype('float32')
        map_y_32 = map_y.astype('float32')
        warped = cv2.remap(self.src_image, map_x_32, map_y_32, cv2.INTER_CUBIC)

        cv2.imwrite("dst-interpolated.jpg", cv2.transpose(warped))

    def unwrap_label_perspective(self, source_map):
        """
        Unwrap label using transform, unlike unwrap_label_interpolation doesn't require scipy
        """
        width, height = self.get_label_size()
        self.dst_image = np.zeros((height, width, 3), np.uint8)

        dx = float(width) / (self.COL_COUNT - 1)
        dy = float(height) / (self.ROW_COUNT - 1)

        dx_int = int(np.ceil(dx))
        dy_int = int(np.ceil(dy))

        for row_index in range(self.ROW_COUNT - 1):
            for col_index in range(self.COL_COUNT - 1):
                src_cell = (source_map[row_index][col_index],
                            source_map[row_index][col_index + 1],
                            source_map[row_index + 1][col_index],
                            source_map[row_index + 1][col_index + 1])

                dst_cell = np.int32([[0, 0], [dx, 0], [0, dy], [dx, dy]])

                M = cv2.getPerspectiveTransform(np.float32(src_cell), np.float32(dst_cell))
                dst = cv2.warpPerspective(self.src_image, M, (dx_int, dy_int))
                x_offset = int(dx * col_index)
                y_offset = int(dy * row_index)

                self.dst_image[y_offset:y_offset + dy_int,
                               x_offset:x_offset + dx_int] = dst

    def get_roi_rect(self, points):
        max_x = min_x = points[0][0]
        max_y = min_y = points[0][1]
        for point in points:
            x, y = point
            if x > max_x:
                max_x = x
            if x < min_x:
                min_x = x
            if y > max_y:
                max_y = y
            if y < min_y:
                min_y = y

        return np.array([
            [min_x, min_y],
            [max_x, min_y],
            [max_x, max_y],
            [min_x, max_y]
        ])

    def get_roi(self, image, points):
        rect = self.get_roi_rect(points)
        return image[np.floor(rect[0][1]):np.ceil(rect[2][1]),
                     np.floor(rect[0][0]):np.ceil(rect[1][0])]

    def calc_source_map(self):
        top_points = self.calc_ellipse_points(self.point_a, self.point_b, self.point_c, self.COL_COUNT)
        bottom_points = self.calc_ellipse_points(self.point_d, self.point_e, self.point_f, self.COL_COUNT)

        rows = []
        for row_index in range(self.ROW_COUNT):
            row = []
            for col_index in range(self.COL_COUNT):
                top_point = top_points[col_index]
                bottom_point = bottom_points[col_index]

                delta = (top_point - bottom_point) / float(self.ROW_COUNT - 1)

                point = top_point - delta * row_index
                row.append(point)
                x, y = map(int, point)

                # cv2.line(self.src_image, (x, y), (x, y), color=self.YELLOW_COLOR, thickness=2)
            rows.append(row)
        return np.array(rows)

    def draw_poly_mask(self, color=WHITE_COLOR):
        cv2.polylines(self.src_image, np.int32([self.points]), 1, color)

    def draw_mask(self, color=WHITE_COLOR):
        cv2.line(self.src_image, tuple(self.point_f.tolist()), tuple(self.point_a.tolist()), color)
        cv2.line(self.src_image, tuple(self.point_c.tolist()), tuple(self.point_d.tolist()), color)

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

        cv2.ellipse(self.src_image, center_point, axis, angle, start_angle, end_angle, color=color)

    def calc_ellipse_points(self, left, top, right, points_count):
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

            # cv2.line(self.src_image, (x, y), (x, y), color=self.YELLOW_COLOR, thickness=3)
            points.append([x, y])

        points.reverse()
        return np.array(points)

    def get_ellipse_point(self, a, b, phi):
        """
        Get ellipse radius in polar coordinates
        """
        return a * np.cos(phi), b * np.sin(phi)

    def get_label_size(self):
        top_left = self.point_a
        top_right = self.point_c
        bottom_right = self.point_d
        bottom_left = self.point_f

        width1 = np.linalg.norm(top_left - top_right)
        width2 = np.linalg.norm(bottom_left - bottom_right)
        avg_width = int((width1 + width2) * np.pi / 4)

        height1 = np.linalg.norm(top_left - bottom_left)
        height2 = np.linalg.norm(top_right - bottom_right)
        avg_height = int((height1 + height2) / 2)
        return avg_width, avg_height


if __name__ == '__main__':
    Main().run()