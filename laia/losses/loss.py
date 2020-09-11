import torch


class Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target, **kwargs):
        raise NotImplementedError
