
import cv2

import logging

from table_recognizer.surname_table_recognizer import SurnameTableRecognizer
from utils.image_utils import binarize_image, canny, align_table
from utils.log_utils import log_image

log = logging.getLogger(__name__)


class Pipeline(object):
    """Common pipeline class fo all pipeline tasks."""

    def __init__(self, source=None):
        self.source = source

    def __iter__(self):
        return self.generator()

    def generator(self):
        """Yields the pipeline data."""

        while self.has_next():
            try:
                data = next(self.source) if self.source else {}
                if self.filter(data):
                    yield self.map(data)
            except StopIteration:
                return

    def __or__(self, other):
        """Allows to connect the pipeline task using | operator."""

        if other is not None:
            other.source = self.generator()
            return other
        else:
            return self

    def filter(self, data):
        """Overwrite to filter out the pipeline data."""

        return True

    def map(self, data):
        """Overwrite to map the pipeline data."""

        return data

    def has_next(self):
        """Overwrite to stop the generator in certain conditions."""

        return True


class ImageToTable(Pipeline):
    def __init__(self):
        super().__init__()
        self.recognizer = SurnameTableRecognizer()

    def map(self, data):
        data['table'] = self.recognizer.recognize(data['img'])
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
