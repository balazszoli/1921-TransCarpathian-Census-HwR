import os
import pickle

import cv2
import scipy.spatial.distance

from pipes.pipline import Pipeline
import numpy as np

import logging

log = logging.getLogger(__name__)


class Matcher(object):

    def __init__(self, vector_size=32, pickled_db_path="features.pck"):
        with open(pickled_db_path, 'rb') as fp:
            self.data = pickle.load(fp)
        self.names = []
        self.matrix = []
        for k, v in self.data.items():
            self.names.append(k)
            self.matrix.append(v)
        self.matrix = np.array(self.matrix)
        self.names = np.array(self.names)

        self.vector_size = vector_size

    def match(self, image_path, topn=5):
        features = extract_features(image_path, self.vector_size)
        img_distances = self._cos_cdist(features)
        # getting top 5 records
        nearest_ids = np.argsort(img_distances)[:topn].tolist()
        nearest_img_paths = self.names[nearest_ids].tolist()

        return nearest_img_paths, img_distances[nearest_ids].tolist()

    def _cos_cdist(self, vector):
        # getting cosine distance between search image and images database
        v = vector.reshape(1, -1)
        return scipy.spatial.distance.cdist(self.matrix, v, 'cosine').reshape(-1)


# Feature extractor
def extract_features(img, vector_size=32):
    try:
        # Using KAZE, cause SIFT, ORB and other was moved to additional module
        # which is adding addtional pain during install
        alg = cv2.KAZE_create()
        # Dinding image keypoints
        kps = alg.detect(img)
        # Getting first 32 of them. 
        # Number of keypoints is varies depend on image size and color pallet
        # Sorting them based on keypoint response value(bigger is better)
        kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
        # computing descriptors vector
        kps, dsc = alg.compute(img, kps)
        # Flatten all of them in one big vector - our feature vector
        dsc = dsc.flatten()
        # Making descriptor of same size
        # Descriptor vector size is 64
        needed_size = (vector_size * 64)
        if dsc.size < needed_size:
            # if we have less the 32 descriptors then just adding zeros at the
            # end of our feature vector
            dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
    except cv2.error as e:
        log.exception(e)
        raise

    return dsc


def batch_extractor(images_path, pickled_db_path="features.pck"):
    files = [os.path.join(images_path, p) for p in sorted(os.listdir(images_path))]

    result = {}
    for f in files:
        print('Extracting features from image %s' % f)
        name = f.split('/')[-1].lower()
        result[name] = extract_features(f)

    # saving all our feature vectors in pickled file
    with open(pickled_db_path, 'w') as fp:
        pickle.dump(result, fp)


class FeatureExtractorPipe(Pipeline):
    def map(self, data):
        data['img_features'] = extract_features(data['img'])
        return data


class ImageFilter(Pipeline):
    def __init__(self, matcher):
        super().__init__()
        self.matcher: Matcher = matcher

    def filter(self, data):
        nearest_img_paths, img_distances = self.matcher.match(data['img'])
        log.debug(f'nearest_img_paths: {nearest_img_paths}')
        log.debug(f'img_distances: {img_distances}')
        return True
