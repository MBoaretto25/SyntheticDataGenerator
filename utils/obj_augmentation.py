"""
Sizes variation = [0.25 , 2]
max number of objects per image = 20

Object Transformations:
    Fliplr
    Rotations = (-20, 20)

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

        self._img_allowed_scales = [0.25, 0.5, .75, 1, 1.25, 1.5, 2]

        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        self._aug_pipe_object = iaa.Sequential(
            [
                iaa.Fliplr(0.5),
                sometimes(iaa.Affine(
                    rotate=(-20, 20),
                )),
                iaa.SomeOf((0, 5),
                           [
                               sometimes(iaa.Superpixels(p_replace=(0.20, 0.5), n_segments=(20, 100))),
                               iaa.OneOf([
                                   iaa.GaussianBlur((0, 3.0)),
                                   iaa.AverageBlur(k=(2, 7)),
                                   iaa.MedianBlur(k=(3, 11)),
                               ]),
                               sometimes(iaa.PerspectiveTransform(scale=(0.025, 0.075))),
                               iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                               iaa.OneOf([
                                   iaa.Dropout((0.01, 0.1), per_channel=0.5),
                                   iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                               ]),
                               iaa.Add((-10, 10), per_channel=0.5),
                               iaa.Multiply((0.5, 1.5), per_channel=0.5),
                               iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
                           ],
                           random_order=True
                           )
            ],
            random_order=True
        )

        self._aug_pipe_sample = iaa.Sequential(
            [
            iaa.SomeOf((0, 3),
                       [
                           sometimes(iaa.Superpixels(p_replace=(0.20, 0.5), n_segments=(20, 100))),
                           iaa.OneOf([
                               iaa.GaussianBlur((0, 3.0)),
                               iaa.AverageBlur(k=(2, 7)),
                               iaa.MedianBlur(k=(3, 11)),
                           ]),
                           iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),

                       ],
                       random_order=True
                       ),
                iaa.Add((-25, 25), per_channel=0.5),
                iaa.Multiply((0.5, 1.5), per_channel=0.5),
                iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
            ],
            random_order=True
        )

    def __call__(self, img, sample=False, *args, **kwargs):
        scale = np.random.choice(self._img_allowed_scales)
        img = cv2.resize(img, None, fx=scale, fy=scale)
        if sample:
            return self._aug_pipe_sample.augment_image(img)
        return self._aug_pipe_object.augment_image(img)
