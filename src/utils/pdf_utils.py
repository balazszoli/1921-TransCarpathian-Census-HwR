import PyPDF2
from PIL.Image import Image


def pdf_to_image(pdf_file, file_name):

    input1 = PyPDF2.PdfFileReader(pdf_file)
    page0 = input1.getPage(0)

    if '/XObject' in page0['/Resources']:
        xObject = page0['/Resources']['/XObject'].getObject()

        for obj in xObject:
            if xObject[obj]['/Subtype'] == '/Image':
                size = (xObject[obj]['/Width'], xObject[obj]['/Height'])
                # TODO acces to private param is bad
                data =xObject[obj]._data
                if xObject[obj]['/ColorSpace'] == '/DeviceRGB':
                    mode = "RGB"
                else:
                    mode = "P"

                if '/Filter' in xObject[obj]:
                    if xObject[obj]['/Filter'] == '/FlateDecode':
                        img = Image.frombytes(mode, size, data)
                        img.save(file_name + ".png")
                        img.close()

                        return file_name + ".png"
                    elif xObject[obj]['/Filter'] == '/DCTDecode':
                        img = open(file_name + ".jpg", "wb")
                        img.write(data)
                        img.close()

                        return file_name + ".jpg"
                    elif xObject[obj]['/Filter'] == '/JPXDecode':
                        img = open(file_name + ".jp2", "wb")
                        img.write(data)
                        img.close()

                        return file_name + ".jp2"
                    elif xObject[obj]['/Filter'] == '/CCITTFaxDecode':
                        img = open(file_name + ".tiff", "wb")
                        img.write(data)
                        img.close()

                        return file_name + ".tiff"
                else:
                    img = Image.frombytes(mode, size, data)
                    img.save(file_name + obj[1:] + ".png")

                    return file_name + ".png"
    else:
        print("No image found.")
