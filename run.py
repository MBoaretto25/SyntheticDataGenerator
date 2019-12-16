import os
import numpy as np

import cv2
from imutils import paths
from tqdm import tqdm

from utils import ObjectInsertion
from utils import ObjectAgumentator


class SyntheticDataGenerator:

    ANNOTATION_FORMAT_STND = "{name},{x1},{y1},{x2},{y2},{label}"
    ANNOTATION_FORMAT_YOLO = "{name},{x1},{y1},{w},{h},{label}"
    ANNOTATION_FORMAT_8PTS = "{name},{x1},{y1},{x2},{y2},{x3},{y3},{x4},{y4},{label}"

    def __init__(self, obj_path="./images/objects", bg_path="./images/backgrounds", img_shape=(1920, 1080)):

        self._obj_path = obj_path
        self._bg_path = bg_path

        self._backgrounds = []
        self._objects = dict()

        self._labels = []

        self._max_number_of_objects = 6
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

    def _generate_annotation(self, img_name, img_centroid, label):
        if self._annotation_style == self.ANNOTATION_FORMAT_STND:
            x1, y1, x2, y2 = self._obj_ins.xywd_2_xyxy(img_centroid)
            annot = self._annotation_style.format(name=img_name, x1=x1, y1=y1, x2=x2, y2=y2, label=label)
        elif self._annotation_style == self.ANNOTATION_FORMAT_YOLO:
            annot = self._annotation_style.format(name=img_name, x=img_centroid.x, y=img_centroid.y,
                                                  w=img_centroid.w, h=img_centroid.h, label=label)
        elif self._annotation_style == self.ANNOTATION_FORMAT_8PTS:
            ## TODO this formats
            pass
        else:
            print("Please provide a valid annotation format!")
            raise NotImplemented
        return annot

    def generate_image(self, sample_id=0):

        n_objects = np.random.randint(self._max_number_of_objects)
        background_image = cv2.imread(np.random.choice(self._backgrounds))
        sample_image = cv2.resize(background_image, self._img_shape)

        annotations = []
        centroid_list = []

        img_name = "img{0:06}.jpg".format(sample_id)

        i=0
        while i <= n_objects:
            image, label = self.retrieve_image()
            if image.shape[0] <= sample_image.shape[0] * 0.1:
                continue
            img_centroid, sample_image = self._obj_ins(sample_image, image, centroid_list)
            centroid_list.append(img_centroid)
            annotation = self._generate_annotation(img_name, img_centroid, label)
            annotations.append(annotation)
            i += 1

        sample_image = self._obj_aug(sample_image, sample=True)

        return sample_image, annotations

    @staticmethod
    def _write_sample(img, annotation, annotation_file, img_location):

        try:
            for ann in annotation:
                img_name = ann.split(",")[0]
                annotation_file.write(ann + "\n")

            cv2.imwrite(os.path.join(img_location, img_name), img)
            return True
        except Exception as e:
            print(e)
        return False

    def generate_dataset(self, n_images=1000, dataset_location="./images/synthetic_dataset/", split=True):
        """
        Generate a dataset of synthetic images based on real images cropped and transformed

        Rules:
        * [x] Always keep an aspect of 16:9 for background images
        * [x] Ignore objects that has height smaller than 10% of the background height
        * [x] Objects cannot overlap (IOU) each other more than 40% of its area
        * [x] Objects cannot be cropped by image margins
        * [x] The transformations must follow a logical pattern that mimics real life conditions
        * [] Only use images that are in good weather/lightning conditions
        * [] Object must has a high variability, since diversity it's an important issue

        :param n_images: dataset size
        :param split: whether or not generate a splitted dataset
        :return: A dict with metadata from the generated dataset
        """
        train_annotations = open(os.path.join(dataset_location, "train.txt"), "w")
        trn_path = os.path.join(dataset_location, "train")
        if not os.path.isdir(trn_path):
            os.mkdir(trn_path)

        if split:
            train_prop, validation_prop, test_prop = 0.7, 0.2, 0.1

            # Test Path
            test_annotations = open(os.path.join(dataset_location, "test.txt"), "w")
            tst_path = os.path.join(dataset_location, "test")
            if not os.path.isdir(tst_path):
                os.mkdir(tst_path)

            # Val Path
            validation_annotations = open(os.path.join(dataset_location, "validation.txt"), "w")
            val_path = os.path.join(dataset_location, "val")
            if not os.path.isdir(val_path):
                os.mkdir(val_path)

        for sample_id in tqdm(range(n_images), desc="Generating Synthetic Dataset!"):
            sample, sample_annotation = self.generate_image(sample_id)

            split_set = 0
            if split:
                split_set = np.random.random()

            # Train
            if split_set <= train_prop:
                ret = self._write_sample(sample, sample_annotation, train_annotations, trn_path)
            # Validation
            if train_prop < split_set <= (train_prop+validation_prop):
                ret = self._write_sample(sample, sample_annotation, validation_annotations, val_path)
            # Test
            if (train_prop+validation_prop) < split_set <= (train_prop+validation_prop+test_prop):
                ret = self._write_sample(sample, sample_annotation, test_annotations, tst_path)

        train_annotations.close()
        try:
            validation_annotations.close()
            test_annotations.close()
        except Exception as e:
            # Ignore
            pass
        print("Done!")


synt = SyntheticDataGenerator()
# bg, ann = synt.generate_image()
synt.generate_dataset(n_images=10000)
# cv2.imshow("bg", cv2.resize(bg, (640, 480)))
# cv2.waitKey(0)
# for a in ann:
#     print(a)
