from abc import abstractmethod, ABC

from table_recognizer.table import Table

from table_recognizer.table_utils import complete_table_from_template, find_table_cells_position
from utils.image_utils import load_image, align_image, binarize_image, Image, canny

import logging

log = logging.getLogger(__name__)

HOUSE_NUMBER_COLUMN = 2
SURNAME_COLUMN = 5


class BaseRecognizer(ABC):
    template = None

    def get_template(self, img: Image):

        log.info('Load template for SurnameTableRecognizer')
        log.info(f'Template file name: {img.file_name}')

        img = self.process_img(img)
        cells = find_table_cells_position(img)

        log.info('Template was loaded')
        log.info('-' * 50)
        return cells

    def process_img(self, img: Image) -> Image:
        # img = binarize_image(img)
        img = canny(img)
        img = align_image(img)
        return img

    @abstractmethod
    def recognize(self, img: Image) -> Table:
        raise NotImplementedError


class SurnameTableRecognizer(BaseRecognizer):
    template_table_name = '../images/table_tamplates/surname.png'

    def __init__(self):
        self.template = self.get_template(load_image(self.template_table_name))

    def recognize(self, img: Image) -> Table:
        log.info(f'Recognize image: {img.file_name}')

        img = self.process_img(img)
        cells = find_table_cells_position(img)
        rows = complete_table_from_template(self.template, cells)

        log.info(f'Image recognized')
        log.info('-' * 50)

        return Table(rows)
