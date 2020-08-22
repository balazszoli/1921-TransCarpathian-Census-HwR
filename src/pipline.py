import os
from os.path import isfile, join

import cv2

import logging

from table_recognizer.surname_table_recognizer import SurnameTableRecognizer
from utils.image_utils import binarize_image, canny, align_table
from utils.log_utils import log_image

log = logging.getLogger(__name__)


def list_files(path, valid_exts=None, level=None):
    # Loop over the input directory structure
    for f in os.listdir(path):
        if isfile(join(path, f)):
            yield join(path, f)


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


class LoadImage(Pipeline):
    def __init__(self, src):
        self.src = src

        super(LoadImage, self).__init__()

    def generator(self):
        image = cv2.imread(self.src, cv2.IMREAD_GRAYSCALE)
        file_name = os.path.basename(self.src)

        data = {
            "img_path": self.src,
            "file_name": file_name,
            "img": image,
        }

        yield self.map(data)


class LoadImages(Pipeline):
    def __init__(self, src):
        self.src = src

        super(LoadImages, self).__init__()

    def generator(self):
        source = list_files(self.src)
        while self.has_next():
            img_path = next(source)
            file_name = os.path.basename(img_path)

            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            data = {
                "img_path": img_path,
                "file_name": file_name,
                "img": image,
            }
            log.debug(f'Load image: "{img_path}"')
            if self.filter(data):
                yield self.map(data)


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
