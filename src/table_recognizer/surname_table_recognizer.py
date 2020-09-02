from abc import abstractmethod, ABC

from table_recognizer.table import Table

from table_recognizer.table_utils import complete_table_from_template, find_table_cells_position
from utils.image_utils import load_image, align_table, binarize_image

import logging

log = logging.getLogger(__name__)

HOUSE_NUMBER_COLUMN = 2
SURNAME_COLUMN = 5


class BaseRecognizer(ABC):
    template = None

    def get_template(self, img):
        img = binarize_image(img)
        img = align_table(img)

        return find_table_cells_position(img)

    @abstractmethod
    def recognize(self, data):
        raise NotImplementedError


class SurnameTableRecognizer(BaseRecognizer):
    template_table_name = '../images/table_tamplates/surname.png'

    def __init__(self):
        self.template = self.get_template(load_image(self.template_table_name))

    def recognize(self, img):
        cells = find_table_cells_position(img)
        rows = complete_table_from_template(self.template, cells)

        return Table(rows)


class SurnameTitleTableRecognizer(BaseRecognizer):
    template_table_name = '../images/table_tamplates/surname.png'

    def __init__(self):
        self.template = self.get_template(load_image(self.template_table_name))

    def recognize(self, img):
        # TODO: load template for title surname table
        cells = find_table_cells_position(img)
        rows = complete_table_from_template(self.template, cells)

        return Table(img, rows)
