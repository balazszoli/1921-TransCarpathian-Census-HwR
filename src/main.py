import logging
import pickle

from image.image_similarity import FeatureExtractorPipe, Matcher, ImageFilter
from pipes.image_loader_pipes import LoadImagesFromWeb, LoadImages
from pipes.image_processing_pipes import ScaleImagePipe, ImageCannyPipe, ImageAlignPipe, ImageToTable, \
    ImageBinarizationPipe
from table_recognizer.surname_table_recognizer import SurnameTableRecognizer

FORMAT = '%(asctime)-15s  %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG)

log = logging.getLogger(__name__)

# BASE_URL = "https://library.hungaricana.hu/en/view/KANepszaml_010_Oroszveg_Kajdano__139_Kajdanove-Kajdano"
BASE_URL = "https://library.hungaricana.hu/en/view/KANepszaml_010_Oroszveg_Beregrakos__145_Rakosin-Beregrakos"

# recognizer = SurnameTableRecognizer()

loader = LoadImagesFromWeb(BASE_URL) \
         | ImageCannyPipe() \
         | ImageAlignPipe() \
         | ImageToTable()

for data in loader:
    log.info('----------------Finish----------------')

#
# # ----------------------------------------------------
# # Create image matcher
# # ----------------------------------------------------
# feature_pipe = LoadImages('../images/surname_tables/') \
#                | ScaleImagePipe(3) \
#                | ImageCannyPipe() \
#                | ImageAlignPipe() \
#                | FeatureExtractorPipe()
#
# result = {}
# for data in feature_pipe:
#     result[data['file_name']] = data['img_features']
#
# with open('features.pck', 'wb') as fp:
#     pickle.dump(result, fp)
#
# surname_tables_matcher = Matcher()
#
# # ----------------------------------------------------
# # Filter out all tables except surname tables
# # ----------------------------------------------------
# pipeline = LoadImages('../images/surname_title_tables/') \
#            | ScaleImagePipe(3) \
#            | ImageCannyPipe() \
#            | ImageAlignPipe() \
#            | ImageFilter(surname_tables_matcher) \
#  \
# # Iterate through pipeline
# for data in pipeline:
#     pass
