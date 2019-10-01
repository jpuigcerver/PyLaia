from torch.nn import BCEWithLogitsLoss


class DortmundBCELoss(BCEWithLogitsLoss):
    def __init__(self):
        super().__init__(reduction="sum")

    def forward(self, output, target):
        loss = super().forward(output, target)
        return loss / output.size(0)
