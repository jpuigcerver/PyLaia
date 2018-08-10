import torch


class Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, targets, **kwargs):
        raise NotImplementedError
