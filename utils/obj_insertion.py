import numpy as np
from collections import namedtuple


class ObjectInsertion:
    """
    This class will handle every object manipulation inside the background image area
    """
    def __init__(self, max_iou=0.4):
        self._max_iou = max_iou
        self._Centroid = namedtuple('Centroid', 'x, y, w, h')

    def _iou(self, image, annotation):
        pass

    def _get_new_object_position(self):
        pass


    def __call__(self, bgi, img, annotations, *args, **kwargs):
        w, h = img.shape[:2]
        image_centroids = self._Centroid(1, 2, 3, 4)
        return image_centroids, bgi
