import cv2

from src.table_recognizer.surname_table_recognizer import SurnameTableRecognizer
from src.table_recognizer.table_registry import register_table, TableConfig

import logging as log

log.basicConfig(level=log.DEBUG)

TABLE_REGISTRY = register_table(
    TableConfig(
        recognizer=SurnameTableRecognizer,
        template_table_name='../images/table_tamplates/surname.png'
    )
)
