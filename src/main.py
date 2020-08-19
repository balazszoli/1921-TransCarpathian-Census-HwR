import logging

from pipline import LoadImages, ImageToTable
from table_recognizer.surname_table_recognizer import SURNAME_COLUMN

FORMAT = '%(asctime)-15s  %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG)

log = logging.getLogger(__name__)

pipeline = LoadImages('../images/surname_tables/') | ImageToTable()

# Iterate through pipeline
for data in pipeline:
    cells = data['table'].column_data(SURNAME_COLUMN)
