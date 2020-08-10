import cv2
import numpy as np

from src.table_recognizer.table_utils import continue_lines, find_line_intersection_points
from src.utils.image_utils import write_lines, write_dots

import logging

log = logging.getLogger(__name__)

def recognize_table(img):
    img_height = img.shape[0]
    img_width = img.shape[1]

    # Run edge detection
    # TODO to achive better result we can adjust Canny parameters
    # or at least make it configurable
    canny_image = cv2.Canny(img, 50, 200, None, 3)

    # Run line detection
    lines = cv2.HoughLinesP(canny_image, 1, np.pi / 180, 50, None, 220, 10)

    # TODO: rotate image is table is not ...
    horizontal, vertical = continue_lines(lines, img_height, img_width)

    cell_intersection_positions = find_line_intersection_points(horizontal, vertical)
    if log.level == logging.DEBUG:
        write_dots(img, lines)

    return cell_intersection_positions


def adjust_table(template, table):
    # check rows
    template_table_top_left_angel = template[0][0]
    top_left_angel = table[0][0]

    # This is the simples solution to adjust table to template
    # For first attemp would be OK :)
    # translation is a geometric transformation that moves every point of a figure or a space
    # by the same distance in a given direction.
    a = top_left_angel[0] - template_table_top_left_angel[0]
    b = top_left_angel[1] - template_table_top_left_angel[1]

    return [[(a + point[0], b + point[1]) for point in row] for row in template]
