from config import IMAGE_LOGGER_OUTPUT, DEBUG
from utils.image_utils import write_lines, write_points, write_cells, write_image, Image


def log_image(img: Image):
    if DEBUG:
        write_image(img.data, 'image_' + img.file_name, IMAGE_LOGGER_OUTPUT)


def log_lines(img: Image, lines):
    if DEBUG:
        write_lines(img.data, lines, 'lines_' + img.file_name, IMAGE_LOGGER_OUTPUT)


def log_points(img: Image, points):
    if DEBUG:
        write_points(img.data, points, 'points_' + img.file_name, IMAGE_LOGGER_OUTPUT)


def log_cells(img: Image, cells):
    if DEBUG:
        write_cells(img.data, cells, 'cells_' + img.file_name, IMAGE_LOGGER_OUTPUT)
