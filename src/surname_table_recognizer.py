from collections import OrderedDict, defaultdict

import cv2
import numpy as np

from src.utils import line_equation_coefficients, line_intersection

HOUSE_NUMBER_COLUMN = 2
SURNAME_COLUMN = 5

class SurnameTableRecognizer:

    def __init__(self, file_name):
        img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        self.img = img
        img_height = img.shape[0]
        img_width = img.shape[1]

        # Run edge detection
        # TODO to achive better result we can adjust Canny parameters
        # or at least make it configurable
        canny_image = cv2.Canny(img, 50, 200, None, 3)
        # Run line detection
        lines = cv2.HoughLinesP(canny_image, 1, np.pi / 180, 50, None, 220, 10)

        horizontal, vertical = self._find_table_lines(lines, img_height, img_width)
        cell_intersection_positions = self._find_table_cell_coordinats(horizontal, vertical)

        rows = defaultdict(list)
        for pos in cell_intersection_positions:
            rows[pos[1]].append(pos)

        self.rows = rows

    def get_column_rows(self, column_num):
        """
        Get list of cells from table

        >>> get_column_rows(3)
        # [[(0,0), (10,0), (0,10), (10,10)], ...]

        Args:
            column_num: the column number

        Returns:
            a list of cells of selected column
        """
        column = []
        for row in self.rows.values():
            column.append(row[column_num:column_num + 2])

        column_cell_position = []
        column_iter = iter(column)
        prev = next(column_iter)

        for row in column:
            column_cell_position.append(row + prev)
            prev = row

        return column_cell_position

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

        return self._reduce_lines(horizontal, img_height), self._reduce_lines(vertical, img_width)

    def _reduce_lines(self, lines, border, threshold=10):

        buckets = []
        prev = next(iter(lines.keys()))
        current_bucket = []

        for key in lines.keys():
            # Check if line is close to image border
            # than it would be the image border line which is not the table part
            # so skip them
            if key - threshold < 0 or key + threshold > border:
                continue

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


SurnameTableRecognizer('../images/test.png')\
    .get_column_rows(SURNAME_COLUMN)
