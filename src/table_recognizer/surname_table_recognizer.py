from abc import abstractmethod

import cv2

from src.table_recognizer.base_recognizer import recognize_table
from src.utils.image_utils import compute_image_skew, rotate_image

import logging

HOUSE_NUMBER_COLUMN = 2
SURNAME_COLUMN = 5


class BaseRecognizer(object):

    @abstractmethod
    def recognize(self):
        pass


class SurnameTableRecognizer(BaseRecognizer):
    name = 'SurnameTableRecognizer'

    def __init__(self):
        pass

    def recognize(self, file_name):
        self.img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        angle = compute_image_skew(self.img)

        self.img = rotate_image(self.img, angle)
        logging.debug(f'The image skew angle is {angle}')

        return recognize_table(self.img)


class Table:

    def __init__(self, img, rows):
        self.img = img
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
        for row in self.rows:
            column.append(row[column_num:column_num + 2])

        column_cell_position = []
        column_iter = iter(column)
        prev = next(column_iter)

        for row in column:
            column_cell_position.append(row + prev)
            prev = row

        return column_cell_position