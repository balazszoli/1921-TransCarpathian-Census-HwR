import cv2

from collections import OrderedDict
from src.table_recognizer.base_recognizer import recognize_table, adjust_table
from src.table_recognizer.surname_table_recognizer import BaseRecognizer, Table
import logging as log

TABLE_REGISTRY = OrderedDict()

class TableConfig(object):

    def __init__(self, recognizer: BaseRecognizer, template_table_name):
        self.recognizer = recognizer
        self.name = recognizer.name
        self.template_table_name = template_table_name
        self.template = recognize_table(cv2.imread(template_table_name, cv2.IMREAD_GRAYSCALE))

    def run(self, filename):
        # Init new instance
        rec = self.recognizer()
        rows = rec.recognize(filename)

        rows = adjust_table(self.template, rows)
        return Table(rec.img, rows)


def register_table(table_config):
    if TABLE_REGISTRY.get(table_config.name, None):
        raise ValueError(f"Table '{table_config.recognizer.name}' is already registered")

    TABLE_REGISTRY[table_config.name] = table_config

    log.debug(f'{table_config.name} was registered')

    return TABLE_REGISTRY

