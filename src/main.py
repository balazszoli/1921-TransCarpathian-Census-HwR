import cv2

from src.table_recognizer.surname_table_recognizer import SurnameTableRecognizer, BaseRecognizer, Table, SURNAME_COLUMN
from src.table_recognizer.table_registry import register_table, TableConfig, TABLE_REGISTRY

import logging

from src.utils.image_utils import write_cells

logging.basicConfig(level=logging.DEBUG)

register_table(
    TableConfig(
        recognizer=SurnameTableRecognizer,
        template_table_name='../images/table_tamplates/surname.png'
    )
)

# JUST FOR TEST recognize some images

recognizer: TableConfig = TABLE_REGISTRY[SurnameTableRecognizer.name]

source_dir = '../images/surname_tables/'
output_dir = '../images/output/'
test_images = ['test.png', 'inangel.png', 'inangel2.png', 'test2.png', 'beregrakos.png']

for file_name in test_images:
    print(source_dir + file_name)
    img = cv2.imread(source_dir + file_name, cv2.IMREAD_GRAYSCALE)
    table: Table = recognizer.run(source_dir + file_name)

    # write_dots(table2.img, [dot for row in table2.rows for dot in row])
    write_cells(table.img, output_dir + file_name, table.get_column_rows(SURNAME_COLUMN))
