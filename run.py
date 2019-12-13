import os
import numpy as np

import cv2
from imutils import paths

from utils import ObjectInsertion
from utils import ObjectAgumentator

import pdb


class SyntheticDataGenerator:

    ANNOTATION_FORMAT_STND = "{name}, {x1}, {y1}, {x2}, {y2}, {label}"
    ANNOTATION_FORMAT_YOLO = "{name}, {x1}, {y1}, {w}, {h}, {label}"
    ANNOTATION_FORMAT_8PTS = "{name}, {x1}, {y1}, {x2}, {y2}, {x3}, {y3}, {x4}, {y4}, {label}"

    def __init__(self, obj_path="./images/objects", bg_path="./images/backgrounds", img_shape = (1920, 1080)):

        self._obj_path = obj_path
        self._bg_path = bg_path

        self._backgrounds = []
        self._objects = dict()

        self._labels = []

        self._max_number_of_objects = 20
        self._img_shape = img_shape

        self._obj_aug = ObjectAgumentator()
        self._obj_ins = ObjectInsertion()

        self._annotation_style = self.ANNOTATION_FORMAT_STND

        self.read_image_paths()

    def read_image_paths(self):
        # Read background images paths
        self._backgrounds = list(paths.list_images(self._bg_path))

        # Read objects images paths
        self._labels = [dirs for _, dirs, _ in os.walk(self._obj_path) if dirs][0]
        for label in self._labels:
            self._objects[label] = list(paths.list_images(os.path.join(self._obj_path, label)))

    def retrieve_image(self):
        label = np.random.choice(self._labels)
        image_dict = self._objects[label]
        random_image_ref = np.random.choice(image_dict)
        image = cv2.imread(random_image_ref)

        t_image = self._obj_aug(image)
        return t_image, label

    def _generate_annotation(self):
        pass

    def generate_image(self):

        n_objects = np.random.randint(self._max_number_of_objects)
        background_image = cv2.imread(np.random.choice(self._backgrounds))
        background_image = cv2.resize(background_image, self._img_shape)

        annotations = []
        centroid_list = []

        i=0
        while i <= n_objects:
            image, label = self.retrieve_image()
            if image.shape[0] <= background_image.shape[0] * 0.1:
                continue
            img_centroid, background_image = self._obj_ins(background_image, image, centroid_list)
            centroid_list.append(img_centroid)
            print(img_centroid)
            if self._annotation_style == self.ANNOTATION_FORMAT_STND:
                ## TODO add image path+id+format
                x1, y1, x2, y2 = self._obj_ins._xywd_2_xyxy(img_centroid)
                annot = self._annotation_style.format(name=i, x1=x1, y1=y1, x2=x2, y2=y2, label=label)
            if self._annotation_style == self.ANNOTATION_FORMAT_YOLO:
                annot = self._annotation_style.format(name=i, x=img_centroid.x, y=img_centroid.y,
                                                      w=img_centroid.w, h=img_centroid.h, label=label)
            annotations.append(annot)
            i += 1

        return background_image, annotations

    def generate_dataset(self, n_images=1000, dataset_location="./images/synthetic_dataset/", split=True):
        """
        Generate a dataset of synthetic images based on real images cropped and transformed

        Rules:
        * Always keep an aspect of 16:9 for background images
        * Ignore objects that has height smaller than 10% of the background height
        * Only use images that are in good weather/lightning conditions
        * Objects cannot overlap (IOU) each other more than 40$ of its area
        * Objects cannot be cropped by image margins
        * The transformations must follow a logical pattern that mimics real life conditions
        * Object must has a high variability, since diversity it's an important issue
        * labels must have the same size of samples

        :param n_images: dataset size
        :param split: whether or not generate a splitted dataset
        :return: A dict with metadata from the generated dataset
        """
        train_annotations = open("train.txt", "w")
        if split:
            train_size, validation_size, test_size = (0.7, 0.2, 0.1)
            validation_annotations = open("validation.txt", "w")
            test_annotations = open("test.txt", "w")

        pass

synt = SyntheticDataGenerator()
bg, ann = synt.generate_image()
cv2.imshow("bg", cv2.resize(bg, (640, 480)))
cv2.waitKey(0)
print(ann)