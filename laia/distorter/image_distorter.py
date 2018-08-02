from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from builtins import range
from functools import reduce

import numpy as np
import torch

try:
    from imgdistort_pytorch import affine, dilate, erode
except ImportError:
    import warnings

    warnings.warn("Missing imgdistort library")

from laia.data import PaddedTensor
from laia.distorter.distorter import Distorter
from laia.utils import image_collage


class ImageDistorter(Distorter):
    DEFAULT_SCALE_PROB = 0.5
    DEFAULT_SCALE_MEAN = 0.0
    DEFAULT_SCALE_PREC = 120.0

    DEFAULT_HSHEAR_PROB = 0.5
    DEFAULT_HSHEAR_MEAN = 0.0
    DEFAULT_HSHEAR_PREC = 4.0

    DEFAULT_VSHEAR_PROB = 0.0
    DEFAULT_VSHEAR_MEAN = 0.0
    DEFAULT_VSHEAR_PREC = 5.0

    DEFAULT_ROTATE_PROB = 0.5
    DEFAULT_ROTATE_MEAN = 0.0
    DEFAULT_ROTATE_PREC = 120.0

    DEFAULT_HTRANSLATE_PROB = 0.5
    DEFAULT_HTRANSLATE_MEAN = 0.0
    DEFAULT_HTRANSLATE_PREC = 2500

    DEFAULT_VTRANSLATE_PROB = 0.5
    DEFAULT_VTRANSLATE_MEAN = 0.0
    DEFAULT_VTRANSLATE_PREC = 1000

    DEFAULT_DILATE_PROB = 0.5
    DEFAULT_DILATE_SRATE = 0.4
    DEFAULT_DILATE_RRATE = 0.8

    DEFAULT_ERODE_PROB = 0.5
    DEFAULT_ERODE_SRATE = 0.4
    DEFAULT_ERODE_RRATE = 0.8

    def __init__(self, **kwargs):
        super(ImageDistorter, self).__init__()

        # If true, assumes that all images in the batch are aligned to their
        # center. Otherwise, it assumes that images are aligned at (0, 0).
        self._aligned_centers = kwargs.get("aligned_centers", False)

        # Scale parameters. Scaling is applied at the center of the image.
        # Log-normal distribution with mean 0, precision = 1 / variance.
        self._scale_prob = kwargs.get("scale_prob", self.DEFAULT_SCALE_PROB)
        self._scale_prec = kwargs.get("scale_prec", self.DEFAULT_SCALE_PREC)
        self._scale_mean = kwargs.get("scale_mean", self.DEFAULT_SCALE_MEAN)

        # Horizontal shear parameters.
        # Normal distribution with mean 0, precision = 1 / variance.
        self._hshear_prob = kwargs.get("hshear_prob", self.DEFAULT_HSHEAR_PROB)
        self._hshear_prec = kwargs.get("hshear_prec", self.DEFAULT_HSHEAR_PREC)
        self._hshear_mean = kwargs.get("hshear_mean", self.DEFAULT_HSHEAR_MEAN)

        # Vertical shear parameters.
        # Normal distribution with mean 0, precision = 1 / variance.
        self._vshear_prob = kwargs.get("vshear_prob", self.DEFAULT_VSHEAR_PROB)
        self._vshear_prec = kwargs.get("vshear_prec", self.DEFAULT_VSHEAR_PREC)
        self._vshear_mean = kwargs.get("vshear_mean", self.DEFAULT_VSHEAR_MEAN)

        # Rotate parameters [relative to the maximum aspect ratio of the image].
        # von Mises distribution with mean 0, precision = 1 / variance.
        self._rotate_prob = kwargs.get("rotate_prob", self.DEFAULT_ROTATE_PROB)
        self._rotate_prec = kwargs.get("rotate_prec", self.DEFAULT_ROTATE_PREC)
        self._rotate_mean = kwargs.get("rotate_mean", self.DEFAULT_ROTATE_MEAN)
        # Map rotation angle to [-pi, pi]
        self._rotate_mean = (self._rotate_mean + math.pi) % (2 * math.pi) - math.pi

        # Translate parameters [relative to the size of each dimension].
        # Normal distribution with mean 0, precision = 1 / variance.
        self._htranslate_prob = kwargs.get(
            "htranslate_prob", self.DEFAULT_HTRANSLATE_PROB
        )
        self._htranslate_prec = kwargs.get(
            "htranslate_prec", self.DEFAULT_HTRANSLATE_PREC
        )
        self._htranslate_mean = kwargs.get(
            "htranslate_mean", self.DEFAULT_HTRANSLATE_MEAN
        )

        self._vtranslate_prob = kwargs.get(
            "vtranslate_prob", self.DEFAULT_VTRANSLATE_PROB
        )
        self._vtranslate_prec = kwargs.get(
            "vtranslate_prec", self.DEFAULT_VTRANSLATE_PREC
        )
        self._vtranslate_mean = kwargs.get(
            "vtranslate_mean", self.DEFAULT_VTRANSLATE_MEAN
        )

        # Dilate parameters.
        # Geometric (kernel size) and Bernoulli (kernel values) distributions.
        # In the Bernoulli distribution, p depends on the distance to the center
        # of the kernel.
        self._dilate_prob = kwargs.get("dilate_prob", self.DEFAULT_DILATE_PROB)
        self._dilate_srate = kwargs.get("dilate_srate", self.DEFAULT_DILATE_SRATE)
        self._dilate_rrate = kwargs.get("dilate_rrate", self.DEFAULT_DILATE_RRATE)

        # Erode parameters.
        # Geometric (kernel size) and Bernoulli (kernel values) distributions.
        # In the Bernoulli distribution, p depends on the distance to the center
        # of the kernel.
        self._erode_prob = kwargs.get("erode_prob", self.DEFAULT_ERODE_PROB)
        self._erode_srate = kwargs.get("erode_srate", self.DEFAULT_ERODE_SRATE)
        self._erode_rrate = kwargs.get("erode_rrate", self.DEFAULT_ERODE_RRATE)

    def __call__(self, x, y=None, destroy_input=False):
        xs = None
        if isinstance(x, PaddedTensor):
            xs = x.sizes.cpu()  # Make sure that the sizes are in the CPU
            x = x.data

        assert torch.is_tensor(x)
        assert x.dim() == 4, "Batch input must be a NxCxHxW tensor"
        assert xs is None or torch.is_tensor(xs)
        assert xs is None or (
            xs.dim() == 2 and xs.size()[1] == 2
        ), "Batch size input must be a Nx2 tensor"
        assert y is None or torch.is_tensor(y)

        # Batch dimensions
        N, C, H, W = x.size()

        # Prepare affine transformations
        M = torch.DoubleTensor(N, 2, 3)
        for n in range(N):
            h = H if xs is None else xs[n, 0]
            w = W if xs is None else xs[n, 1]
            cy = 0.5 * H if self._aligned_centers else h * 0.5
            cx = 0.5 * W if self._aligned_centers else w * 0.5
            m = self.__sample_affine_matrix(w, h, cx, cy)[:2, :]
            M[n].copy_(m)

        dilate_kernel = []
        erode_kernel = []
        for n in range(N):
            sd = self.__sample_structuring_element(
                self._dilate_prob, self._dilate_srate, self._dilate_rrate
            )
            se = self.__sample_structuring_element(
                self._erode_prob, self._erode_srate, self._erode_rrate
            )
            dilate_kernel.append(sd.view(sd.numel()))
            erode_kernel.append(se.view(se.numel()))

        # Copy affine and structure matrices to the appropriate gpu, if necessary
        if x.is_cuda:
            M = M.cuda(x.get_device())
            dilate_kernel = [k.cuda(x.get_device()) for k in dilate_kernel]
            erode_kernel = [k.cuda(x.get_device()) for k in erode_kernel]

        y = affine(x, M, y)

        tmp = x if destroy_input else x.clone()
        tmp = dilate(x, dilate_kernel, tmp)

        y = erode(tmp, erode_kernel, y)

        if xs is None:
            return y
        else:
            if y.is_cuda:
                xs = xs.cuda(y.get_device())
            return PaddedTensor(y, xs)

    def __sample_affine_matrix(self, w, h, cx=None, cy=None):
        if cx is None:
            cx = w * 0.5
        if cy is None:
            cy = h * 0.5

        # Output affine matrix
        m = torch.eye(3).double()

        # 1. Translation (including centering the image)
        if torch.rand(1)[0] < self._htranslate_prob:
            d = self._htranslate_mean + (
                torch.randn(1)[0] / math.sqrt(self._htranslate_prec)
            )
            m[0, 2] = d * w - cx
        else:
            m[0, 2] = -cx

        if torch.rand(1)[0] < self._vtranslate_prob:
            d = self._vtranslate_mean + (
                torch.randn(1)[0] / math.sqrt(self._vtranslate_prec)
            )
            m[1, 2] = d * h - cy
        else:
            m[1, 2] = -cy

        # 2. Horizontal and vertical shearing.
        if self._hshear_prec > 0.0 or self._vshear_prob > 0.0:
            if torch.rand(1)[0] < self._hshear_prob:
                hs = self._hshear_mean + (
                    torch.randn(1)[0] / math.sqrt(self._hshear_prec)
                )
            else:
                hs = 0.0
            if torch.rand(1)[0] < self._vshear_prob:
                vs = self._vshear_mean + (
                    torch.randn(1)[0] / math.sqrt(self._vshear_prec)
                )
            else:
                vs = 0.0
            d = torch.DoubleTensor([[1.0, hs, 0.0], [vs, 1.0, 0.0], [0.0, 0.0, 1.0]])
            m = torch.mm(d, m)

        # 3. Rotation.
        if torch.rand(1)[0] < self._rotate_prob:
            R = max(w / h, h / w)
            a = np.random.vonmises(self._rotate_mean, self._rotate_prec * R)
            d = torch.DoubleTensor(
                [
                    [math.cos(a), -math.sin(a), 0.0],
                    [math.sin(a), math.cos(a), 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
            m = torch.mm(d, m)

        # 4. Scaling
        if torch.rand(1)[0] < self._scale_prob:
            s = math.exp(
                self._scale_mean + (torch.randn(1)[0] / math.sqrt(self._scale_prec))
            )
            d = torch.DoubleTensor([[s, 0.0, 0.0], [0.0, s, 0.0], [0.0, 0.0, 1.0]])
            m = torch.mm(d, m)

        # 5. Translate back to the original center
        d = torch.DoubleTensor([[1.0, 0.0, +cx], [0.0, 1.0, +cy], [0.0, 0.0, 1.0]])
        m = torch.mm(d, m)
        return m

    @staticmethod
    def __sample_structuring_element(p, srate, rrate):
        if torch.rand(1)[0] < p:
            # Sample size (height and width) of the structuring element.
            # Only the following height/width sizes are available.
            Sv = [3, 5, 7, 9, 11, 13, 15]
            Sp = [srate * math.pow(1 - srate, x - Sv[0]) for x in Sv]
            SpSum = sum(Sp)
            Sp = [x / SpSum for x in Sp]
            Mh, Mw = np.random.choice(Sv, size=2, p=Sp)
            S = torch.ByteTensor(int(Mh), int(Mw)).zero_()
            for y in range(Mh):
                for x in range(Mw):
                    dy = y - Mh // 2
                    dx = x - Mw // 2
                    r = math.sqrt(dx * dx + dy * dy)
                    if torch.rand(1)[0] < math.exp(-rrate * r):
                        S[y, x] = 1
        else:
            # This structuring element matrix preserves the image.
            S = torch.ByteTensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        return S


if __name__ == "__main__":
    from laia.common.arguments import add_argument, args, add_defaults
    import argparse
    from PIL import Image, ImageOps

    add_defaults("seed")
    add_argument("--draw_boundary", action="store_true")
    add_argument("--aligned_center", action="store_true")
    add_argument("--ncol", type=int, default=1)
    add_argument("--scale", type=float, default=1.0)
    add_argument("--gpu", type=int, default=0)
    add_argument("img", type=argparse.FileType("r"), nargs="*")
    args = args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    xlist = []
    xsize = []
    for img in args.img:
        im = Image.open(img)
        im = ImageOps.invert(im).convert("L")

        x = torch.from_numpy(np.asarray(im))
        if x.dim() != 3:
            x.unsqueeze_(0)
        xlist.append(x)
        xsize.append((x.size()[1], x.size()[2]))

    # Create batch tensor
    bs = reduce(lambda acc, x: (max(acc[0], x[0]), max(acc[1], x[1])), xsize, (0, 0))
    N, C, H, W = len(xlist), 1, bs[0], bs[1]
    batch = torch.Tensor().resize_(N, C, H, W)
    batch_size = torch.IntTensor(xsize)

    if args.gpu > 0:
        batch = batch.cuda(args.gpu - 1)

    # Copy all images to the batch tensor
    if args.aligned_center:
        for i, (x, xs) in enumerate(zip(xlist, xsize)):
            dy = (H - xs[0]) // 2
            dx = (W - xs[1]) // 2
            batch[i, :, dy : (dy + xs[0]), dx : (dx + xs[1])].copy_(x)
    else:
        for i, (x, xs) in enumerate(zip(xlist, xsize)):
            batch[i, :, : xs[0], : xs[1]].copy_(x)

    # Distort batch
    distorter = ImageDistorter(erode_prob=1.0)
    batch2 = distorter(PaddedTensor(batch, batch_size)).data

    image_collage(
        batch,
        batch_size,
        scale=args.scale,
        ncol=args.ncol,
        draw_boundary=args.draw_boundary,
    ).show()
    image_collage(
        batch2,
        batch_size,
        scale=args.scale,
        ncol=args.ncol,
        draw_boundary=args.draw_boundary,
    ).show()
