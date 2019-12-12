"""
Sizes variation = [0.25 , 2]
max number of objects per image = 20

Object Transformations:
    Fliplr
    Rotations = (-20, 20)

    PiecewiseAffine = scale(0.02, 0.075)
    PerspectiveTransform = scale(.025, 0.075)
    MaxPooling k = 2
    MotionBlur

    segmentation = n_segments = 100, p_replace=0.5

Image Transformations :
    AdditiveGaussianNoise
    Multiply
    Dropout

    HistogramEqualization

    LogContrast
    GaussianBlur
"""
import cv2
import numpy as np
from imgaug import augmenters as iaa


class ObjectAgumentator:

    def __init__(self):

        self._img_allowed_scales = [.5, .75, 1, 1.25, 1.5, 2, 2.25]

        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        self._aug_pipe = iaa.Sequential(
            [
                iaa.Fliplr(0.5), # horizontally flip 50% of all images
                sometimes(iaa.Affine(
                    rotate=(-20, 20),  # rotate by -45 to +45 degrees
                )),
                iaa.SomeOf((0, 5),
                           [
                               sometimes(iaa.Superpixels(p_replace=(0.20, 0.5), n_segments=(20, 100))),
                               iaa.OneOf([
                                   iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
                                   iaa.AverageBlur(k=(2, 7)),  # blur image using local means (kernel sizes between 2 and 7)
                                   iaa.MedianBlur(k=(3, 11)),  # blur image using local medians (kernel sizes between 2 and 7)
                               ]),
                               iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),  # add gaussian noise
                               iaa.OneOf([
                                   iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
                                   iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                               ]),
                               iaa.Add((-10, 10), per_channel=0.5),  # change brightness of images
                               iaa.Multiply((0.5, 1.5), per_channel=0.5),  # change brightness of images
                               iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),  # improve or worsen the contrast
                           ],
                           random_order=True
                           )
            ],
            random_order=True
        )

    def __call__(self, img, *args, **kwargs):
        scale = np.random.choice(self._img_allowed_scales)
        print(scale)
        img = cv2.resize(img, None, fx=scale, fy=scale)
        return self._aug_pipe.augment_image(img)
