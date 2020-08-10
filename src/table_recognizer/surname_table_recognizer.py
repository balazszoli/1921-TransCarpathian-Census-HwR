from abc import abstractmethod, ABC

import cv2

from src.table_recognizer.base_recognizer import recognize_table, adjust_table
from src.utils.image_utils import compute_image_skew, rotate_image

import logging

log = logging.getLogger(__name__)

HOUSE_NUMBER_COLUMN = 2
SURNAME_COLUMN = 5

class BaseRecognizer(ABC):
    template = None

    def recognize(self, file_name):
        log.debug(f'Load image: {file_name}')
        img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)

        angle = compute_image_skew(img)
        log.debug(f'The image skew angle is {angle}')

        img = rotate_image(img, angle)

        rows = recognize_table(img)
        rows = adjust_table(self.get_template(), rows)

        return Table(img, rows)

    @abstractmethod
    def get_template(self):
        pass


class SurnameTableRecognizer(BaseRecognizer):
    name = 'SurnameTableRecognizer'
    template_table_name = '../images/table_tamplates/surname.png'

    def get_template(self):
        if self.template is None:
            self.template = recognize_table(cv2.imread(self.template_table_name, cv2.IMREAD_GRAYSCALE))
        return self.template


class SurnameTitleTableRecognizer(BaseRecognizer):
    name = 'SurnameTableRecognizer'
    template_table_name = '../images/table_tamplates/surname.png'

    def get_template(self):
        if self.template is None:
            self.template = recognize_table(cv2.imread(self.template_table_name, cv2.IMREAD_GRAYSCALE))


class Table:

    def __init__(self, img, rows):
        self.img = img
        self.rows = rows

    def get_column_data(self, column_num):
        """
        Return only data rows from table.

        @param column_num: column number
        @return: return rows without headers
        """
        rows = self.get_column_rows(column_num)

        return rows[3:]

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