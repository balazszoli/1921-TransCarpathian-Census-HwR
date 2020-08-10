import logging

from table_recognizer.surname_table_recognizer import SurnameTableRecognizer, HOUSE_NUMBER_COLUMN
from utils.image_utils import write_cells

FORMAT = '%(asctime)-15s  %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG)

log = logging.getLogger(__name__)


# Just for test
recog = SurnameTableRecognizer()
table = recog.recognize('../images/surname_tables/test2.png')
write_cells(table.img, table.get_column_data(HOUSE_NUMBER_COLUMN))