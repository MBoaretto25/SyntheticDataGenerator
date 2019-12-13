import numpy as np
from collections import namedtuple


class ObjectInsertion:
    """
    This class will handle every object manipulation inside the background image area
    """
    def __init__(self, max_iou=0.4):
        self._max_iou = max_iou
        self._Centroid = namedtuple('Centroid', 'x, y, w, h')

    def _iou(self, obj_a, obj_b):

        # determine the (x, y)-coordinates of the intersection rectangle
        x_a = max(obj_a[0], obj_b[0])
        y_a = max(obj_a[1], obj_b[1])
        x_b = min(obj_a[2], obj_b[2])
        y_b = min(obj_a[3], obj_b[3])

        # compute the area of intersection rectangle
        inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        box_a_area = (obj_a[2] - obj_a[0] + 1) * (obj_a[3] - obj_a[1] + 1)
        box_b_area = (obj_b[2] - obj_b[0] + 1) * (obj_b[3] - obj_b[1] + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = inter_area / float(box_a_area + box_b_area - inter_area)

        # return the intersection over union value
        return iou

    @staticmethod
    def _get_random_point_within_boundaries(position_range, boundary):
        print("position_range, boundary", position_range, boundary)
        return np.random.randint(boundary, position_range - boundary)

    @staticmethod
    def _xywd_2_xyxy(obj):
        x1 = obj.x - obj.w//2
        x2 = obj.x + obj.w//2
        y1 = obj.y - obj.h//2
        y2 = obj.y + obj.h//2
        return [x1, y1, x2, y2]

    def _get_new_object_position(self, bgi_shape, img_shape):
        background_height, background_width = bgi_shape[:2]
        # Define x/y boundaries given the object size
        x_boundary = (img_shape[1] // 2) + 10
        y_boundary = (img_shape[0] // 2) + 10
        x_candidate = self._get_random_point_within_boundaries(background_width, x_boundary)
        y_candidate = self._get_random_point_within_boundaries(background_height, y_boundary)
        return self._Centroid(x_candidate, y_candidate, img_shape[1], img_shape[0])

    def __call__(self, bgi, img, centroid_list, *args, **kwargs):

        while True:
            img_centroid = self._get_new_object_position(bgi.shape, img.shape)
            img_coordinates = self._xywd_2_xyxy(img_centroid)
            is_feasible = not any([self._iou(img_coordinates, self._xywd_2_xyxy(centroid)) > 0.4
                                   for centroid in centroid_list])
            if is_feasible:
                break
        print(img_coordinates[0], img_coordinates[0]+img.shape[0])
        print(img_coordinates[1], img_coordinates[1]+img.shape[1])
        print(img.shape)
        bgi[img_coordinates[1]:img_coordinates[1]+img.shape[0], img_coordinates[0]:img_coordinates[0]+img.shape[1], ...] = img
        return img_centroid, bgi
