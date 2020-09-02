import cv2

from pipes.pipline import Pipeline
from table_recognizer.table_utils import find_table_cells_position, find_lines, continue_lines, \
    find_line_intersection_points

from utils.image_utils import binarize_image, canny, align_table
from utils.log_utils import log_image, log_points, log_lines

import logging

log = logging.getLogger(__name__)


class ImageToTable(Pipeline):
    def __init__(self):
        super().__init__()

    def map(self, data):
        lines = find_lines(data['img'])
        if lines is None or len(lines) == 0:
            log.error(f'Cant recognize any lines in file: {data["file_name"]}')
            return data

        log_lines(data['img'], data['file_name'], lines)

        horizontal, vertical = continue_lines(data['img'], lines)
        log_lines(data['img'], '_horizontal_vertical_' + data['file_name'], horizontal + vertical)

        points = find_line_intersection_points(horizontal, vertical)
        log_points(data['img'], data['file_name'], points)

        return data


class ScaleImagePipe(Pipeline):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def map(self, data):
        height, width = data['img'].shape
        new_height, new_width = int(width / self.scale), int(height / self.scale)

        data['img'] = cv2.resize(data['img'], (new_height, new_width))
        log.debug(f'Scaled from: "{(height, width)}" to: "{(new_height, new_width)}"')
        log_image(data['img'], '_scaled_' + data['file_name'] )
        return data


class ImageBinarizationPipe(Pipeline):
    def map(self, data):
        data['img'] = binarize_image(data['img'])
        log_image(data['img'], data['file_name'] + '_binarized.png')
        return data


class ImageCannyPipe(Pipeline):
    def map(self, data):
        data['img'] = canny(data['img'])
        log_image(data['img'], data['file_name'] + '_canny.png')

        return data


class ImageAlignPipe(Pipeline):
    def map(self, data):
        data['img'] = align_table(data['img'])
        return data
