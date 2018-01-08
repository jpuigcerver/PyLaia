from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import torch

from .distorter import Distorter
from ..data import PaddedTensor
from imgdistort_pytorch import affine, dilate, erode

class ImageDistorter(Distorter):
    def __init__(self, **kwargs):
        super(ImageDistorter, self).__init__()

        # If true, assumes that all images in the batch are aligned to their
        # center. Otherwise, it assumes that images are aligned at (0, 0).
        self._aligned_centers = kwargs.get('aligned_centers', False)

        # Scale parameters. Scaling is applied at the center of the image.
        # Log-normal distribution with mean 0, precision = 1 / variance.
        self._scale_prob = kwargs.get('scale_prob', 0.5)
        self._scale_prec = kwargs.get('scale_prec', 100)
        self._scale_mean = kwargs.get('scale_mean', 0.0)

        # Horizontal shear parameters.
        # Normal distribution with mean 0, precision = 1 / variance.
        self._hshear_prob = kwargs.get('hshear_prob', 0.5)
        self._hshear_prec = kwargs.get('hshear_prec', 4.0)
        self._hshear_mean = kwargs.get('hshear_mean', 0.0)

        # Vertical shear parameters.
        # Normal distribution with mean 0, precision = 1 / variance.
        self._vshear_prob = kwargs.get('vshear_prob', 0.0)
        self._vshear_prec = kwargs.get('vshear_prec', 5)
        self._vshear_mean = kwargs.get('vshear_mean', 0.0)

        # Rotate parameters [relative to the maximum aspect ratio of the image].
        # von Mises distribution with mean 0, precision = 1 / variance.
        self._rotate_prob = kwargs.get('rotate_prob', 0.5)
        self._rotate_prec = kwargs.get('rotate_prec', 120)
        self._rotate_mean = kwargs.get('rotate_mean', 0.0)
        # Map rotation angle to [-pi, pi]
        self._rotate_mean = (
            (self._rotate_mean + math.pi) % (2 * math.pi) - math.pi)

        # Translate parameters [relative to the size of each dimension].
        # Normal distribution with mean 0, precision = 1 / variance.
        self._translate_prob = kwargs.get('translate_prob', 0.5)
        self._translate_prec = kwargs.get('translate_prec', 2500)
        self._translate_mean = kwargs.get('translate_mean', 0.0)

        # Dilate parameters.
        # Geometric (kernel size) and Bernoulli (kernel values) distributions.
        # In the Bernoulli distribution, p depends on the distance to the center
        # of the kernel.
        self._dilate_prob = kwargs.get('dilate_prob', 0.5)
        self._dilate_srate = kwargs.get('dilate_srate', 0.4)
        self._dilate_rrate = kwargs.get('dilate_rrate', 1.0)

        # Erode parameters.
        # Geometric (kernel size) and Bernoulli (kernel values) distributions.
        # In the Bernoulli distribution, p depends on the distance to the center
        # of the kernel.
        self._erode_prob = kwargs.get('erode_prob', 0.5)
        self._erode_srate = kwargs.get('erode_srate', 0.8)
        self._erode_rrate = kwargs.get('erode_rrate', 1.2)

        self._M = None

    def __call__(self, x, y=None):
        xs = None
        if isinstance(x, PaddedTensor):
            xs = x.sizes
            x = x.data

        assert torch.is_tensor(x)
        assert x.dim() == 4, ('Batch input must be a NxCxHxW tensor')
        assert xs is None or torch.is_tensor(xs)
        assert xs is None or (xs.dim() == 2 and xs.size()[1] == 2), (
            'Batch size input must be a Nx2 tensor')
        assert y is None or torch.is_tensor(y)

        # Prepare output
        y = x.clone().zero_() if y is None else y
        if xs is not None:
            ys = xs.cpu()
        else:
            ys = None

        # Batch dimensions
        N, C, H, W = x.size()

        # Prepare affine transformations
        if self._M is None:
            self._M = torch.DoubleTensor(N, 2, 3)
        else:
            self._M.resize_(N, 2, 3)
        for n in xrange(N):
            h = H if ys is None else ys[n, 0]
            w = W if ys is None else ys[n, 1]
            cy = 0.5 * H if self._aligned_centers else h * 0.5
            cx = 0.5 * W if self._aligned_centers else w * 0.5
            m = self.__sample_affine_matrix(w, h, cx, cy)[:2, :]
            self._M[n].copy_(m)

        dilate_kernel = []
        erode_kernel = []
        dilate_kernel_sizes = []
        erode_kernel_sizes = []
        for n in xrange(N):
            sd = self.__sample_structuring_element(
                self._dilate_prob, self._dilate_srate, self._dilate_rrate)
            se = self.__sample_structuring_element(
                self._erode_prob, self._erode_srate, self._erode_rrate)
            dilate_kernel_sizes.append(sd.size())
            erode_kernel_sizes.append(se.size())
            dilate_kernel.append(sd.view(sd.numel()))
            erode_kernel.append(se.view(se.numel()))
        dilate_kernel = torch.cat(dilate_kernel)
        erode_kernel = torch.cat(erode_kernel)
        dilate_kernel_sizes = torch.IntTensor(dilate_kernel_sizes).cpu()
        erode_kernel_sizes = torch.IntTensor(erode_kernel_sizes).cpu()

        affine(x, self._M, y)
        y2 = y.clone().zero_()
        dilate(y, erode_kernel, erode_kernel_sizes, y2)
        erode(y2, dilate_kernel, dilate_kernel_sizes, y)

        if xs is None:
            return y
        else:
            return PaddedTensor(data=y, sizes=ys)

    def __sample_affine_matrix(self, w, h, cx=None, cy=None):
        if cx is None:
            cx = w * 0.5
        if cy is None:
            cy = h * 0.5

        # Output affine matrix
        m = torch.eye(3).double()

        # 1. Translation (including centering the image)
        if torch.rand(1)[0] < self._translate_prob:
            d = self._translate_mean + (
                torch.randn(2) / math.sqrt(self._translate_prec))
            m[0, 2] = d[0] * w - cx
            m[1, 2] = d[1] * h - cy
        else:
            m[0, 2] = -cx
            m[1, 2] = -cy

        # 2. Horizontal and vertical shearing.
        if self._hshear_prec > 0.0 or self._vshear_prob > 0.0:
            if torch.rand(1)[0] < self._hshear_prob:
                hs = self._hshear_mean + (
                    torch.randn(1)[0] / math.sqrt(self._hshear_prec))
            else:
                hs = 0.0
            if torch.rand(1)[0] < self._vshear_prob:
                vs = self._vshear_mean + (
                    torch.randn(1)[0] / math.sqrt(self._vshear_prec))
            else:
                vs = 0.0
            d = torch.DoubleTensor([[1.0,  hs, 0.0],
                                    [ vs, 1.0, 0.0],
                                    [0.0, 0.0, 1.0]])
            m = torch.mm(d, m)


        # 3. Rotation.
        if torch.rand(1)[0] < self._rotate_prob:
            R = max(w / h, h / w)
            a = np.random.vonmises(self._rotate_mean, self._rotate_prec * R)
            d = torch.DoubleTensor([[math.cos(a), -math.sin(a), 0.0],
                                    [math.sin(a),  math.cos(a), 0.0],
                                    [        0.0,          0.0, 1.0]])
            m = torch.mm(d, m)

        # 4. Scaling
        if torch.rand(1)[0] < self._scale_prob:
            s = math.exp(self._scale_mean + (
                torch.randn(1)[0] / math.sqrt(self._scale_prec)))
            d = torch.DoubleTensor([[  s, 0.0, 0.0],
                                    [0.0,   s, 0.0],
                                    [0.0, 0.0, 1.0]])
            m = torch.mm(d, m)

        # 5. Translate back to the original center
        d = torch.DoubleTensor([[1.0, 0.0, +cx],
                                [0.0, 1.0, +cy],
                                [0.0, 0.0, 1.0]])
        m = torch.mm(d, m)
        return m

    def __sample_structuring_element(self, p, srate, rrate):
        if torch.rand(1)[0] < p:
            # Sample size (height and width) of the structuring element.
            # Only the following height/width sizes are available.
            Sv = [3, 5, 7, 9, 11, 13, 15]
            Sp = map(lambda x: srate * math.pow(1 - srate, x - Sv[0]), Sv)
            SpSum = sum(Sp)
            Sp = map(lambda x: x / SpSum, Sp)
            Mh, Mw = np.random.choice(Sv, size=2, p=Sp)
            S = torch.ByteTensor(Mh, Mw).zero_()
            for y in xrange(Mh):
                for x in xrange(Mw):
                    dy = y - Mh // 2
                    dx = x - Mw // 2
                    r = math.sqrt(dx * dx + dy * dy)
                    if torch.rand(1)[0] < math.exp(-rrate * r):
                        S[y, x] = 1
        else:
            # This structuring element matrix preserves the image.
            S = torch.ByteTensor([[0, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, 0]])
        return S


if __name__ == '__main__':
    import argparse
    import sys
    from PIL import Image, ImageOps

    parser = argparse.ArgumentParser()
    parser.add_argument('--center_batch', action='store_true')
    parser.add_argument('--seed', type=int, default=0x12345)
    parser.add_argument('--ncol', type=int, default=1)
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('img', type=argparse.FileType('r'), nargs='*')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    xlist = []
    xsize = []
    for img in args.img:
        im = Image.open(img)
        im = ImageOps.invert(im)

        x = torch.from_numpy(np.asarray(im))
        if x.dim() != 3:
            x.unsqueeze_(0)
        xlist.append(x)
        xsize.append((x.size()[1], x.size()[2]))

    # Create batch tensor
    bs = reduce(lambda acc, x: (max(acc[0], x[0]), max(acc[1], x[1])),
                xsize, (0, 0))
    N, C, H, W = len(xlist), 1, bs[0], bs[1]
    batch = torch.Tensor().resize_(N, C, H, W)

    batch_size = torch.IntTensor(xsize)

    if args.center_batch:
        for i, (x, xs) in enumerate(zip(xlist, xsize)):
            dy = (H - xs[0]) // 2
            dx = (W - xs[1]) // 2
            batch[i, :, dy:(dy+xs[0]), dx:(dx+xs[1])].copy_(x)
    else:
        for i, (x, xs) in enumerate(zip(xlist, xsize)):
            batch[i, :, :xs[0], :xs[1]].copy_(x)

    dist = ImageDistorter()
    batch2 = dist(PaddedTensor(data=batch, sizes=batch_size)).data

    def collate(x):
        N, C, H, W = x.size()
        nrow = int(math.ceil(N / args.ncol))
        im = torch.Tensor(H * nrow, W * args.ncol)
        n = 0
        for r in xrange(nrow):
            for c in xrange(args.ncol):
                im[(r * H):(r * H + H), (c * W):(c * W + W)].copy_(
                    x[n, 0, :, :])
                n = n + 1

        im = Image.fromarray(im.numpy())
        if args.scale != 1.0:
            W = int(math.ceil(im.size[0] * args.scale))
            H = int(math.ceil(im.size[1] * args.scale))
            im = im.resize((H, W))
        return im

    collate(batch).show()
    collate(batch2).show()
