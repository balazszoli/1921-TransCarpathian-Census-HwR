import logging
import pickle

from image.image_similarity import batch_extractor, FeatureExtractorPipe, Matcher, ImageFilter
from pipline import LoadImages, ImageToTable, LoadImage, ImageBinarizationPipe, ScaleImagePipe, ImageCannyPipe, \
    ImageAlignPipe
from table_recognizer.surname_table_recognizer import SURNAME_COLUMN

FORMAT = '%(asctime)-15s  %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG)

log = logging.getLogger(__name__)

# ----------------------------------------------------
# Create image matcher
# ----------------------------------------------------
feature_pipe = LoadImages('../images/surname_tables/') \
               | ScaleImagePipe(16) \
               | ImageCannyPipe() \
               | ImageAlignPipe() \
               | FeatureExtractorPipe()

result = {}
for data in feature_pipe:
    result[data['file_name']] = data['img_features']

with open('features.pck', 'wb') as fp:
    pickle.dump(result, fp)

surname_tables_matcher = Matcher()
# ----------------------------------------------------

pipeline = LoadImages('../images/surname_tables/') \
           | ScaleImagePipe(16) \
           | ImageCannyPipe() \
           | ImageAlignPipe() \
           | ImageFilter(surname_tables_matcher) \
 \
# Iterate through pipeline
for data in pipeline:
    pass
