from typing import Callable, Optional, Union

import numpy as np
import pytorch_lightning as pl
from tqdm.auto import tqdm

from laia.decoders import CTCGreedyDecoder
from laia.utils import SymbolsTable


def compute_word_prob(symbols, hyp, prob, input_separator):
    # compute mean confidence score and mean confidence score by word
    space_id = symbols._sym2val[input_separator]
    word_prob_list = []
    word_prob, word_chars = [], ""
    for value, prob in zip(hyp, prob):
        char = symbols[value]
        if value != space_id:
            word_prob.append(prob)
            word_chars += char
        elif word_chars:
            word_prob_list.append(sum(word_prob) / len(word_prob))
            word_prob, word_chars = [], ""
    if word_chars:
        word_prob_list.append(sum(word_prob) / len(word_prob))
    return word_prob_list


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
        print_line_confidence_scores: bool = False,
        print_word_confidence_scores: bool = False,
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
        self.print_line_confidence_scores = print_line_confidence_scores
        self.print_word_confidence_scores = print_word_confidence_scores

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, *args):
        super().on_test_batch_end(trainer, pl_module, outputs, batch, *args)
        img_ids = pl_module.batch_id_fn(batch)
        hyps = self.decoder(outputs)["hyp"]
        probs = self.decoder(outputs)["prob-htr-char"]

        # compute mean confidence score
        line_probs = [np.mean(prob) for prob in probs]

        # compute mean confidence score by word
        word_probs = [
            compute_word_prob(self.syms, hyps[i], probs[i], self.input_space)
            for i in range(len(probs))
        ]
        for i, (img_id, hyp, line_prob, word_prob) in enumerate(
            zip(img_ids, hyps, line_probs, word_probs)
        ):

            if self.use_symbols:
                hyp = [self.syms[v] for v in hyp]
                if self.convert_spaces:
                    hyp = [
                        self.output_space if sym == self.input_space else sym
                        for sym in hyp
                    ]
            if self.join_string is not None:
                hyp = self.join_string.join(str(x) for x in hyp)

            if self.print_line_confidence_scores:
                self.write(
                    f"{img_id}{self.separator}{line_prob:.2f}{self.separator}{hyp}"
                    if self.include_img_ids
                    else f"{line_prob:.2f}{self.separator}{hyp}"
                )

            elif self.print_word_confidence_scores:
                word_prob = [f"{prob:.2f}" for prob in word_prob]
                self.write(
                    f"{img_id}{self.separator}{word_prob}{self.separator}{hyp}"
                    if self.include_img_ids
                    else f"{word_prob}{self.separator}{hyp}"
                )

            else:
                self.write(
                    f"{img_id}{self.separator}{hyp}"
                    if self.include_img_ids
                    else str(hyp)
                )

    def write(self, value):
        # no idea why adding the line break is necessary. in distributed mode with
        # >1 gpus some lines will not include it. also happens with print() so it's
        # not a tqdm issue. couldn't reproduce it on toy examples but it does
        # happen in the iam-htr example
        return tqdm.write(value + "\n", end="")
