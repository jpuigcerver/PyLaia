from typing import Any, Callable, Optional

import torch

from laia.decoders import CTCGreedyDecoder
from laia.engine import EvaluatorModule


class HTREvaluatorModule(EvaluatorModule):
    def __init__(
        self,
        model: torch.nn.Module,
        batch_input_fn: Optional[Callable] = None,
        batch_id_fn: Optional[Callable] = None,
        decoder: Optional[Callable] = CTCGreedyDecoder(),
        segmentation: bool = False,
    ):
        super().__init__(model, batch_input_fn=batch_input_fn, batch_id_fn=batch_id_fn)
        self.decoder = decoder
        self.segmentation = segmentation

    def test_step(self, batch: Any, batch_idx: int):
        super().test_step(batch, batch_idx)
        self.decoder(self.batch_y_hat, segmentation=self.segmentation)
