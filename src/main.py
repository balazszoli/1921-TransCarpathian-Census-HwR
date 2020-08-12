import logging

from os import listdir
from os.path import isfile, join

import cv2

from table_recognizer.surname_table_recognizer import SurnameTableRecognizer, HOUSE_NUMBER_COLUMN, \
    SurnameTitleTableRecognizer
from utils.image_utils import write_cells

FORMAT = '%(asctime)-15s  %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG)

log = logging.getLogger(__name__)

# -----------------------------------
# Just for test
# -----------------------------------

recog = SurnameTableRecognizer()

# load all images from surname_tables dir
files = [f for f in listdir('../images/surname_tables/') if isfile(join('../images/surname_tables/', f))]

# extract house numbers from surname tables
OUTPUT_DIR = '../images/output'
for file_name in files:
    table = recog.recognize('../images/surname_tables/' + file_name )

    cells = table.column_data(HOUSE_NUMBER_COLUMN)
    for cell in cells:
        crop_img = table.img[cell[-2][1]:cell[1][1], cell[-2][0]:cell[1][0]]

        # write house numbers as separate images
        cv2.imwrite(f"{OUTPUT_DIR}/{str(cell)}_{file_name}_cells.jpg", crop_img.copy())

    write_cells(table.img, file_name, table.column_data(HOUSE_NUMBER_COLUMN))
