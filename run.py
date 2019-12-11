import os
import random

import cv2
from imutils import paths


class SyntheticDataGenerator:

    ANNOTATION_FORMAT = "{name}, {x1}, {y1}, {x2}, {y2}, {label}"

    def __init__(self, obj_path="./images/objects", bg_path="./images/backgrounds", img_shape = (1920, 1080)):

        self._obj_path = obj_path
        self._bg_path = bg_path

        self._backgrounds = []
        self._objects = dict()

        self._labels = []

        self._max_number_of_objects = 20
        self._img_shape = img_shape

        self.read_images()

    def read_image_paths(self):
        # Read background images paths
        self._backgrounds = list(paths.list_images(self._bg_path))

        # Read objects images paths
        self._labels = [dirs for _, dirs, _ in os.walk(self._obj_path) if dirs][0]
        for label in self._labels:
            self._objects['label'] = list(paths.list_images(os.path.join(self._obj_path, label)))

    def generate_image(self):

        n_images = random.randint(self._max_number_of_objects)
        background_path = self._backgrounds[random.randint(len(self._backgrounds))]

        background_image = cv2.imread(background_path)
        background_image = cv2.resize(background_image, self._img_shape)

        annotation = []

        for i in range(n_images):

            pass

        return annotation

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
