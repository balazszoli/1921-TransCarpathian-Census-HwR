import logging
import os
import tempfile
from os.path import isfile, join
from urllib.error import HTTPError
from urllib.request import urlopen
from pathlib import Path

import cv2

from pipes.pipline import Pipeline
from utils.pdf_utils import pdf_to_image

log = logging.getLogger(__name__)


def list_files(path, valid_exts=None, level=None):
    # Loop over the input directory structure
    for f in os.listdir(path):
        if isfile(join(path, f)):
            yield join(path, f)


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


DOWNLOADS_IMAGE_DIR = '../images/downloads/'
BASE_URL = "https://library.hungaricana.hu/en/view/KANepszaml_010_Oroszveg_Kajdano__139_Kajdanove-Kajdano"


class LoadImagesFromWeb(Pipeline):
    def __init__(self, base_url):
        super().__init__()
        self.page_count = 1
        self.base_url = base_url

    def generator(self):
        while True:
            try:
                pdf_access_url = self.get_pdf_access_url(self.base_url)
                log.info(f'PDF access URL is: {pdf_access_url}')
            except HTTPError:
                raise

            page = "page{:04d}.pdf".format(self.page_count)
            log.info(f'Current page: {page}')

            pdf_file = self.load_pdf_to_temp_file(pdf_access_url, page)

            place_name = self.base_url.split('/')[-1]
            self.mkdir(DOWNLOADS_IMAGE_DIR + place_name)

            img_path = pdf_to_image(pdf_file, DOWNLOADS_IMAGE_DIR + place_name + '/' + page)
            log.info(f'Image was saved to: {img_path}')

            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            data = {
                "img_path": img_path,
                "file_name": img_path.split('/')[-1],
                "img": image,
            }

            if self.filter(data):
                yield self.map(data)

            self.page_count += 1


    def get_pdf_access_url(self, page_url):
        html = urlopen(page_url).read().decode("utf-8")

        start_index = html.find('"pdfAccessUrl": "') + len('"pdfAccessUrl": "')
        end_index = html[start_index:].find('",') + start_index

        pdf_access_url = html[start_index:end_index]

        if not pdf_access_url:
            raise ValueError(f'invalid pdf_access_url for: {page_url}')

        return pdf_access_url

    def mkdir(self, dir):
        Path(dir).mkdir(parents=True, exist_ok=True)

    def load_pdf_to_temp_file(self, pdf_access_url, file_name):
        pdf = urlopen(pdf_access_url + file_name).read()
        pdf_file = tempfile.TemporaryFile()
        pdf_file.write(pdf)
        pdf_file.seek(0)

        return pdf_file