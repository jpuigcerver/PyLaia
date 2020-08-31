from typing import Any, Callable, Dict, Iterable, Optional

import pytorch_lightning as pl
import torch

from laia.callbacks.meters import SequenceError, char_to_word_seq
from laia.decoders import CTCGreedyDecoder
from laia.engine import EngineModule
from laia.losses import CTCLoss


class HTREngineModule(EngineModule):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: str,
        optimizer_kwargs: Dict,
        delimiters: Iterable,
        criterion: Optional[Callable] = CTCLoss(),
        monitor: str = "va_cer",
        batch_input_fn: Optional[Callable] = None,
        batch_target_fn: Optional[Callable] = None,
        batch_id_fn: Optional[Callable] = None,
    ):
        super().__init__(
            model,
            optimizer,
            optimizer_kwargs,
            criterion,
            monitor=monitor,
            batch_input_fn=batch_input_fn,
            batch_target_fn=batch_target_fn,
            batch_id_fn=batch_id_fn,
        )
        self.delimiters = delimiters
        self.decoder = CTCGreedyDecoder()
        self.tr_cer = SequenceError()
        self.tr_wer = SequenceError()
        self.va_cer = SequenceError()
        self.va_wer = SequenceError()

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self.tr_cer.reset()
        self.tr_wer.reset()

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self.va_cer.reset()
        self.va_wer.reset()

    def training_step(self, batch: Any, batch_idx: int) -> pl.TrainResult:
        result = super().training_step(batch, batch_idx)
        _, batch_y = self.prepare_batch(batch)
        batch_decode = self.decoder(self.batch_y_hat)
        # cer
        self.tr_cer.add(batch_y, batch_decode)
        result.log(
            "tr_cer",
            torch.tensor(self.tr_cer.value),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        # wer
        batch_decode_words = [
            char_to_word_seq(b, self.delimiters) for b in batch_decode
        ]
        batch_y_words = [char_to_word_seq(b, self.delimiters) for b in batch_y]
        self.tr_wer.add(batch_y_words, batch_decode_words)
        result.log(
            "tr_wer",
            torch.tensor(self.tr_wer.value),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return result

    def validation_step(self, batch: Any, batch_idx: int) -> Optional[pl.EvalResult]:
        result = super().validation_step(batch, batch_idx)
        _, batch_y = self.prepare_batch(batch)
        batch_decode = self.decoder(self.batch_y_hat)
        # cer
        self.va_cer.add(batch_y, batch_decode)
        cer_value = torch.tensor(self.va_cer.value)
        result.log(
            "va_cer",
            cer_value,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        if self.monitor == "va_cer":
            result.early_stop_on = cer_value
            result.checkpoint_on = cer_value
        # wer
        batch_decode_words = [
            char_to_word_seq(b, self.delimiters) for b in batch_decode
        ]
        batch_y_words = [char_to_word_seq(b, self.delimiters) for b in batch_y]
        self.va_wer.add(batch_y_words, batch_decode_words)
        wer_value = torch.tensor(self.va_wer.value)
        result.log(
            "va_wer",
            wer_value,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        if self.monitor == "va_wer":
            result.early_stop_on = wer_value
            result.checkpoint_on = wer_value
        return result
