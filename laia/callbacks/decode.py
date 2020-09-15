import sys
from typing import Optional, Union

import pytorch_lightning as pl
from tqdm.auto import tqdm

from laia.utils import SymbolsTable


class Decode(pl.callbacks.Callback):
    def __init__(
        self,
        syms: Union[dict, SymbolsTable],
        use_symbols: bool = False,
        input_space: str = "<space>",
        output_space: str = "",
        convert_spaces: bool = False,
        join_str: Optional[str] = None,
        separator: str = " ",
        print_img_ids: bool = True,
    ):
        super().__init__()
        self.syms = syms
        self.use_symbols = use_symbols
        self.input_space = input_space
        self.output_space = output_space
        self.convert_spaces = convert_spaces
        self.join_str = join_str
        self.separator = separator
        self.print_img_ids = print_img_ids

    def on_test_batch_end(self, trainer, pl_module, batch, *args):
        super().on_test_batch_end(trainer, pl_module, batch, *args)
        img_ids = pl_module.batch_id_fn(batch)
        decoder = pl_module.decoder
        for i, (img_id, out) in enumerate(zip(img_ids, decoder.output)):
            if self.use_symbols:
                out = [self.syms[v] for v in out]
            if self.convert_spaces:
                out = [
                    self.output_space if sym == self.input_space else sym for sym in out
                ]
            if self.join_str is not None:
                out = self.join_str.join(str(x) for x in out)
            tqdm.write(
                f"{img_id}{self.separator}{out}" if self.print_img_ids else out,
                file=sys.stdout,
            )
