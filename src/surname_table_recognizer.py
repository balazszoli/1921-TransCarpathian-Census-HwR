from collections import OrderedDict

import cv2
import numpy as np

from src.utils import line_equation_coefficients, line_intersection


class SurnameTableRecognizer:
    def recognize(self, file_name):
        img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        img_height = img.shape[0]
        img_width = img.shape[1]

        # Run edge detection
        # TODO to achive better result we can adjust Canny parameters
        # or at least make it configurable
        canny_image = cv2.Canny(img, 50, 200, None, 3)
        # Run line detection
        lines = cv2.HoughLinesP(canny_image, 1, np.pi / 180, 50, None, 220, 10)

        horizontal, vertical = self._find_table_lines(lines, img_height, img_width)
        cell_positions = self._find_table_cell_coordinats(horizontal, vertical)

    def _find_table_lines(self, lines, img_height, img_width):
        horizontal = dict()
        vertical = dict()
        for i in range(0, len(lines)):
            line = lines[i][0]
            angel = np.arctan2(line[3] - line[1], line[2] - line[0]) * 180. / np.pi

            if -91 < angel < -89:
                vertical[line[0]] = [line[0], 0, line[2], img_height]
            elif angel == 0:
                horizontal[line[1]] = [0, line[1], img_width, line[3]]

        horizontal = OrderedDict(sorted(horizontal.items()))
        vertical = OrderedDict(sorted(vertical.items()))

        return self._reduce_lines(horizontal), self._reduce_lines(vertical)

    def _reduce_lines(self, lines, threshold=10):

        buckets = []
        prev = next(iter(lines.keys()))
        current_bucket = []

        for key in lines.keys():
            if key > threshold + prev:
                if len(current_bucket) == 0:
                    buckets.append([prev])
                buckets.append(current_bucket)
                current_bucket = []

            current_bucket.append(key)
            prev = key

        if len(current_bucket) > 0:
            buckets.append(current_bucket)

        result = []

        for bucket in buckets:
            mean = np.mean(bucket)
            result.append(lines[min(bucket, key=lambda x: abs(x - mean))])

        return result

    def _find_table_cell_coordinats(self, horizontal, vertical):
        dots = []

        # Find intersection of horizontal and vertical lines and plot them on image (just for observation)
        for hor_line in horizontal:
            L1 = line_equation_coefficients(hor_line[:2], hor_line[2:])

            for ver_line in vertical:

                L2 = line_equation_coefficients(ver_line[:2], ver_line[2:])
                inter = line_intersection(L1, L2)
                if inter:
                    dots.append(inter)

        return dots
