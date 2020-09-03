import logging
import os
import time
from pathlib import Path
from urllib.request import urlopen
from bs4 import BeautifulSoup
import tempfile

from utils.pdf_utils import pdf_to_image

FORMAT = '%(asctime)-15s  %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG)
log = logging.getLogger(__name__)

HOME = 'https://library.hungaricana.hu'
TITLE_PAGE = '/en/collection/KarpataljaiNepszamlalas1921/'


def download_training_images():
    html = urlopen(HOME + TITLE_PAGE).read().decode("utf-8")
    soup = BeautifulSoup(html, 'html.parser')

    card_footer = soup.find(class_='card-footer')
    district_links = card_footer.find_all('a')

    for district_link in district_links:
        district_href = district_link.get('href').strip()

        html = urlopen(HOME + district_href).read().decode("utf-8")
        soup = BeautifulSoup(html, 'html.parser')

        card_footer = soup.find(class_='card-footer')
        city_links = card_footer.find_all('a')

        log.info(f'------------ Start {district_href} ------------')
        for city_link in city_links:
            city_href = city_link.get('href').strip()
            log.info('Process:' + city_href)

            if 'collection' in city_href:
                log.info('*** This is a city link')
                continue

            pdf_access_url = get_pdf_access_url(HOME + city_href)
            city = city_href.split('/')[-2]

            def save_image(page, folder):
                log.info(f'Download pdf: {pdf_access_url + page}')

                pdf = urlopen(pdf_access_url + page).read()
                pdf_file = tempfile.TemporaryFile()
                pdf_file.write(pdf)
                pdf_file.seek(0)

                filename = pdf_to_image(pdf_file, f'./train/{folder}/{city}-{page}')
                log.info(f'File {filename} was saved')

            # Save first page as title table
            page = "page{:04d}.pdf".format(0)
            if not os.path.exists(f'./train/table1/{city}-{page}.jpg'):
                save_image(page, 'table1')
                time.sleep(1)
            else:
                log.info(f'Already exist: {city}-{page}.jpg')

            # Save second page as simple table
            page = "page{:04d}.pdf".format(1)
            if not os.path.exists(f'./train/table2/{city}-{page}.jpg'):
                save_image(page, 'table2')
                time.sleep(1)
            else:
                log.info(f'Already exist: {city}-{page}.jpg')


def get_pdf_access_url(page_url):
    html = urlopen(page_url).read().decode("utf-8")

    start_index = html.find('"pdfAccessUrl": "') + len('"pdfAccessUrl": "')
    end_index = html[start_index:].find('",') + start_index

    pdf_access_url = html[start_index:end_index]

    if not pdf_access_url:
        raise ValueError(f'invalid pdf_access_url for: {page_url}')

    log.info(f'PDF access URL: {pdf_access_url}')

    return pdf_access_url


def mkdir(dir):
    Path(dir).mkdir(parents=True, exist_ok=True)


if __name__ == '__main__':
    mkdir('./train/table1')
    mkdir('./train/table2')
    download_training_images()
