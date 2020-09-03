import logging

from table_recognizer.surname_table_recognizer import SurnameTableRecognizer
from utils.image_utils import load_image, binarize_image, align_image, write_cells

FORMAT = '%(asctime)-15s  %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG)

log = logging.getLogger(__name__)


if __name__ == '__main__':

    rec = SurnameTableRecognizer()

    img = load_image('../images/surname_tables/test.png')
    table = rec.recognize(img)

    write_cells(img, table.rows)

