import logging

from table_recognizer.surname_table_recognizer import SurnameTableRecognizer, HOUSE_NUMBER_COLUMN, \
    SurnameTitleTableRecognizer
from utils.image_utils import write_cells

FORMAT = '%(asctime)-15s  %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG)

log = logging.getLogger(__name__)


# Just for test
recog = SurnameTableRecognizer()
table = recog.recognize('../images/surname_tables/test2.png')

write_cells(table.img, table.column_data(HOUSE_NUMBER_COLUMN))


# recog = SurnameTitleTableRecognizer()
# table = recog.recognize('../images/surname_title_tables/test.png')
# write_cells(table.img, table.column_data(HOUSE_NUMBER_COLUMN))