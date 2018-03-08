from __future__ import absolute_import
from __future__ import division

import cv2
import math
import numpy as np
import operator
import torch

from collections import OrderedDict
from laia.losses.loss import Loss
from laia.nn.temporal_pyramid_maxpool_2d import TemporalPyramidMaxPool2d
from functools import reduce
from torch.nn.functional import binary_cross_entropy_with_logits
from PIL import ImageOps


def build_dortmund_model(phoc_size, levels=5):
    model = torch.nn.Sequential(OrderedDict([
        # conv1_1
        ('conv1_1', torch.nn.Conv2d(1, 64, kernel_size=3, padding=1)),
        ('relu1_1', torch.nn.ReLU(inplace=True)),
        # conv1_2
        ('conv1_2', torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)),
        ('relu1_2', torch.nn.ReLU(inplace=True)),
        ('maxpool1', torch.nn.MaxPool2d(2)),
        # conv2_1
        ('conv2_1', torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)),
        ('relu2_1', torch.nn.ReLU(inplace=True)),
        # conv2_2
        ('conv2_2', torch.nn.Conv2d(128, 128, kernel_size=3, padding=1)),
        ('relu2_2', torch.nn.ReLU(inplace=True)),
        ('maxpool2', torch.nn.MaxPool2d(2)),
        # conv3_1
        ('conv3_1', torch.nn.Conv2d(128, 256, kernel_size=3, padding=1)),
        ('relu3_1', torch.nn.ReLU(inplace=True)),
        # conv3_2
        ('conv3_2', torch.nn.Conv2d(256, 256, kernel_size=3, padding=1)),
        ('relu3_2', torch.nn.ReLU(inplace=True)),
        # conv3_3
        ('conv3_3', torch.nn.Conv2d(256, 256, kernel_size=3, padding=1)),
        ('relu3_3', torch.nn.ReLU(inplace=True)),
        # conv3_4
        ('conv3_4', torch.nn.Conv2d(256, 256, kernel_size=3, padding=1)),
        ('relu3_4', torch.nn.ReLU(inplace=True)),
        # conv3_5
        ('conv3_5', torch.nn.Conv2d(256, 256, kernel_size=3, padding=1)),
        ('relu3_5', torch.nn.ReLU(inplace=True)),
        # conv3_6
        ('conv3_6', torch.nn.Conv2d(256, 256, kernel_size=3, padding=1)),
        ('relu3_6', torch.nn.ReLU(inplace=True)),
        # conv4_1
        ('conv4_1', torch.nn.Conv2d(256, 512, kernel_size=3, padding=1)),
        ('relu4_1', torch.nn.ReLU(inplace=True)),
        # conv4_2
        ('conv4_2', torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)),
        ('relu4_2', torch.nn.ReLU(inplace=True)),
        # conv4_3
        ('conv4_3', torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)),
        ('relu4_3', torch.nn.ReLU(inplace=True)),
        # SPP layer
        ('tpp5', TemporalPyramidMaxPool2d(levels=levels)),
        # Linear layers
        ('fc6', torch.nn.Linear(512 * sum(range(1, levels + 1)), 4096)),
        ('relu6', torch.nn.ReLU(inplace=True)),
        ('drop6', torch.nn.Dropout()),
        ('fc7', torch.nn.Linear(4096, 4096)),
        ('relu7', torch.nn.ReLU(inplace=True)),
        ('drop7', torch.nn.Dropout()),
        ('fc8', torch.nn.Linear(4096, phoc_size)),
        # Predicted PHOC
        # ('sigmoid', torch.nn.Sigmoid())
    ]))

    # Initialize parameters as Caffe does
    for name, param in model.named_parameters():
        if name[-5:] == '.bias':
            # Initialize bias to 0
            param.data.fill_(0)
        else:
            # compute fan in
            fan_in = reduce(operator.mul, param.data.size()[1:])
            param.data.normal_(mean=0, std=math.sqrt(2.0 / fan_in))

    return model


def dortmund_distort(img, random_limits=(0.8, 1.1)):
    """
    Creates an augmentation by computing a homography from three points in the
    image to three randomly generated points.
    """
    y, x = img.shape[:2]
    src_point = np.float32([[x / 2, y / 3],
                            [2 * x / 3, 2 * y / 3],
                            [x / 3, 2 * y / 3]])
    random_shift = (np.random.rand(3, 2) - 0.5) * 2 * (
                random_limits[1] - random_limits[0]) / 2 + np.mean(
        random_limits)
    dst_point = src_point * random_shift.astype(np.float32)
    transform = cv2.getAffineTransform(src_point, dst_point)
    if img.ndim == 3:
        border_value = np.median(
            np.reshape(img, (img.shape[0] * img.shape[1], -1)), axis=0)
    else:
        border_value = float(np.median(img))
    return cv2.warpAffine(img, transform, dsize=(x, y),
                          borderValue=border_value)


class DortmundImageToTensor(object):
    def __call__(self, x):
        x = x.convert('L')
        x = ImageOps.invert(x)
        x = np.asarray(x, dtype=np.float32)
        x = dortmund_distort(x / 255.0)
        if x.shape != 3:
            x = np.expand_dims(x, axis=-1)
        x = np.transpose(x, (2, 0, 1))
        return torch.from_numpy(x)


class DortmundBCELoss(Loss):
    def __call__(self, output, target):
        loss = binary_cross_entropy_with_logits(output, target,
                                                size_average=False)
        loss = loss / output.size(0)
        if output.grad is not None:
            output.grad = output.grad / output.size(0)
        return loss
