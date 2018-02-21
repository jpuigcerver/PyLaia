from __future__ import division

import math
from builtins import range

import numpy as np
import torch
from PIL import Image


def image_collage(x, xs=None, scale=1.0, ncol=1, draw_boundary=False):
    assert torch.is_tensor(x)
    assert xs is None or torch.is_tensor(xs)
    x = x.cpu()
    if xs is not None:
        xs = xs.cpu()

    N, C, H, W = x.size()
    nrow = math.ceil(N / ncol)
    im = x.new(3, H * nrow, W * ncol)
    n = 0
    for r in range(nrow):
        for c in range(ncol):
            im[:, (r * H):(r * H + H), (c * W):(c * W + W)].copy_(x[n, :, :, :])
            n += 1

    if draw_boundary and xs is not None:
        magenta = x.new([255, 0, 255])
        n = 0
        for r in range(nrow):
            for c in range(ncol):
                h, w = xs[n, 0], xs[n, 1]
                im[:, (r * H):(r * H + h), (c * W)].copy_(magenta.view(3, 1))
                im[:, (r * H), (c * W):(c * W + w)].copy_(magenta.view(3, 1))
                im[:, (r * H):(r * H + h), (c * W + w - 1)].copy_(magenta.view(3, 1))
                im[:, (r * H + h - 1), (c * W):(c * W + w)].copy_(magenta.view(3, 1))
                n += 1

    im = im.permute(1, 2, 0).contiguous()
    im = Image.fromarray(np.uint8(im.numpy()))

    if scale != 1.0:
        H = math.ceil(im.size[0] * scale)
        W = math.ceil(im.size[1] * scale)
        im = im.resize((H, W), resample=Image.BICUBIC)

    return im
