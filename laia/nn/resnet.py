import torch.nn as nn
import torch.nn.functional as F

from laia.data import PaddedTensor


class ResnetConv2dBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, batch_norm=True):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.conv2 = nn.Conv2d(
            out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        if batch_norm:
            self.bn1 = nn.BatchNorm2d(out_planes)
            self.bn2 = nn.BatchNorm2d(out_planes)
        else:
            self.bn1 = lambda x: x
            self.bn2 = lambda x: x

        self.stride = stride
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, out_planes, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_planes),
            )
        else:
            self.shortcut = lambda x: x

    def forward(self, x):
        if isinstance(x, PaddedTensor):
            x, xs = x.data, x.sizes
        else:
            xs = None
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        y += self.shortcut(x)
        y = F.relu(y)
        if xs is not None:
            ys = (xs + self.stride - 1) / self.stride
            y = PaddedTensor(y, ys)
        return y
