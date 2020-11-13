import sys
from typing import Callable, Optional, Union

import pytorch_lightning as pl
from tqdm.auto import tqdm

from laia.decoders import CTCGreedyDecoder
from laia.utils import SymbolsTable


class Decode(pl.Callback):
    def __init__(
        self,
        decoder: Optional[Callable] = CTCGreedyDecoder(),
        syms: Optional[Union[dict, SymbolsTable]] = None,
        use_symbols: bool = False,
        input_space: str = "<space>",
        output_space: str = " ",
        convert_spaces: bool = False,
        join_string: Optional[str] = None,
        separator: str = " ",
        include_img_ids: bool = True,
    ):
        super().__init__()
        self.decoder = decoder
        self.syms = syms
        self.use_symbols = use_symbols
        if use_symbols:
            assert syms is not None
        self.input_space = input_space
        self.output_space = output_space
        self.convert_spaces = convert_spaces
        if convert_spaces:
            assert use_symbols
        self.join_string = join_string
        self.separator = separator
        self.include_img_ids = include_img_ids

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, *args):
        super().on_test_batch_end(trainer, pl_module, outputs, batch, *args)
        img_ids = pl_module.batch_id_fn(batch)
        hyps = self.decoder(outputs)["hyp"]
        for i, (img_id, hyp) in enumerate(zip(img_ids, hyps)):
            if self.use_symbols:
                hyp = [self.syms[v] for v in hyp]
                if self.convert_spaces:
                    hyp = [
                        self.output_space if sym == self.input_space else sym
                        for sym in hyp
                    ]
            if self.join_string is not None:
                hyp = self.join_string.join(str(x) for x in hyp)
            self.write(
                f"{img_id}{self.separator}{hyp}" if self.include_img_ids else str(hyp)
            )

    def write(self, value):
        return tqdm.write(value, file=sys.stdout)
