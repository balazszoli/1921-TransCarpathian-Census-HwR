import logging

import PyPDF2
from PIL.Image import Image
from urllib.request import urlopen

log = logging.getLogger(__name__)


def get_pdf_access_url(page_url):
    """
    To download pdf we should have pdfAccessUrl which are part of javascript code
    which runs on web page. In our case we just extract it and then use

    @param page_url: source page url
    @return: pdf access url
    """
    html = urlopen(page_url).read().decode("utf-8")

    start_index = html.find('"pdfAccessUrl": "') + len('"pdfAccessUrl": "')
    end_index = html[start_index:].find('",') + start_index

    pdf_access_url = html[start_index:end_index]

    if not pdf_access_url:
        raise ValueError(f'invalid pdf_access_url for: {page_url}')

    log.info(f'PDF access URL: {pdf_access_url}')

    return pdf_access_url


def pdf_to_image(pdf_file, destination_file_name):
    """
    Extract images from pdf
    
    @param pdf_file: source pdf file
    @param destination_file_name: where to save the image
    @return: image name with extension

    """
    input1 = PyPDF2.PdfFileReader(pdf_file)
    page0 = input1.getPage(0)

    if '/XObject' in page0['/Resources']:
        xObject = page0['/Resources']['/XObject'].getObject()

        for obj in xObject:
            if xObject[obj]['/Subtype'] == '/Image':
                size = (xObject[obj]['/Width'], xObject[obj]['/Height'])
                # TODO acces to private param is bad
                data = xObject[obj]._data
                if xObject[obj]['/ColorSpace'] == '/DeviceRGB':
                    mode = "RGB"
                else:
                    mode = "P"

                if '/Filter' in xObject[obj]:
                    if xObject[obj]['/Filter'] == '/FlateDecode':
                        img = Image.frombytes(mode, size, data)
                        img.save(destination_file_name + ".png")
                        img.close()

                        return destination_file_name + ".png"
                    elif xObject[obj]['/Filter'] == '/DCTDecode':
                        img = open(destination_file_name + ".jpg", "wb")
                        img.write(data)
                        img.close()

                        return destination_file_name + ".jpg"
                    elif xObject[obj]['/Filter'] == '/JPXDecode':
                        img = open(destination_file_name + ".jp2", "wb")
                        img.write(data)
                        img.close()

                        return destination_file_name + ".jp2"
                    elif xObject[obj]['/Filter'] == '/CCITTFaxDecode':
                        img = open(destination_file_name + ".tiff", "wb")
                        img.write(data)
                        img.close()

                        return destination_file_name + ".tiff"
                else:
                    img = Image.frombytes(mode, size, data)
                    img.save(destination_file_name + obj[1:] + ".png")

                    return destination_file_name + ".png"
    else:
        print("No image found.")
