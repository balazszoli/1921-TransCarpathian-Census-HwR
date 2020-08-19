from config import IMAGE_LOGGER_OUTPUT, DEBUG
from utils.image_utils import write_lines, write_points, write_cells, write_image


def log_image(img, file_name):
    if DEBUG:
        write_image(img, 'image_' + file_name, IMAGE_LOGGER_OUTPUT)


def log_lines(img, file_name, lines):
    if DEBUG:
        write_lines(img, lines, 'lines_' + file_name, IMAGE_LOGGER_OUTPUT)


def log_points(img, file_name, points):
    if DEBUG:
        write_points(img, points, 'points_' + file_name, IMAGE_LOGGER_OUTPUT)


def log_cells(img, file_name, cells):
    if DEBUG:
        write_cells(img, cells, 'cells_' + file_name, IMAGE_LOGGER_OUTPUT)
