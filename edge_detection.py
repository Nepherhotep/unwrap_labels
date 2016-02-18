import cv2
import numpy as np
from matplotlib import pyplot as plt


class EdgeDetector(object):
    BLUR = 3

    def __init__(self, src_image, pixel_points=[], percent_points=[]):
        """
        Points define

        A         B
        |         |
        |         |
        |         |
        |         |
        D         C

        :param src_image:
        :param pixel_points:
        :param percent_points:
        :return:
        """
        self.src_image = src_image
        self.preprocessed_image = None
        self.width = self.src_image.shape[1]
        self.height = src_image.shape[0]
        self.points = pixel_points
        self.percent_points = percent_points

        self.point_a, self.point_b, self.point_c, self.point_d = (None, None, None, None)

    def load_points(self):
        if not self.points:
            points = []
            for point in self.percent_points:
                x = int((point[0] + 0.5) * self.width)
                y = int((0.5 - point[1]) * self.height)
                points.append((x, y))

            self.points = np.array(points)

        if not len(self.points) == 4:
            raise ValueError('Array of points should have length == 4')

        self.point_a, self.point_b, self.point_c, self.point_d = self.points

    def detect_edges(self):
        gray = cv2.cvtColor(self.src_image, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(gray, (self.BLUR, self.BLUR), 0)

        scale = 1
        delta = 0
        ddepth = cv2.CV_16S

        # Gradient-X
        grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=self.BLUR, scale=scale, delta=delta,
                           borderType=cv2.BORDER_DEFAULT)

        # Gradient-Y
        grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=self.BLUR, scale=scale, delta=delta,
                           borderType=cv2.BORDER_DEFAULT)

        # converting back to uint8
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)

        # join to images into as single one
        dst = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        cv2.imwrite('edges.jpg', dst)
        
        cv2.destroyAllWindows()

    def detect(self):
        self.load_points()
        self.detect_edges()


if __name__ == '__main__':
    imcv = cv2.imread('image2.jpg', cv2.IMREAD_UNCHANGED)
    points = [[-0.45846, +0.21444],
              [+0.46992, +0.19538],
              [+0.46483, -0.16038],
              [-0.45507, -0.13624]]

    d = EdgeDetector(imcv, percent_points=points)
    d.detect()
