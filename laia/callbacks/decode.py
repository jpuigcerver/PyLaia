from typing import Callable, Optional, Union

import numpy as np
import pytorch_lightning as pl
from tqdm.auto import tqdm

from laia.decoders import CTCGreedyDecoder
from laia.utils import SymbolsTable


def get_slices(space_indexes, max_len):
    slices = [slice(0, space_indexes[0])]                               # first slice 0:index_1
    for i in range(0, len(space_indexes)-1):
        slices.append(slice(space_indexes[i]+1, space_indexes[i+1]))    # other slices index_{n}+1:index_{n+1}
    slices.append(slice(space_indexes[-1]+1, max_len))                  # last slice index_{-1}:-1
    return slices


def compute_word_prob(hyp, prob, space_token):
    space_index = np.argwhere(np.array(hyp) == space_token).reshape(-1).tolist()        # [4 16]
    if space_index:
        word_slices = get_slices(space_index, len(hyp))      # [(0, 4), (5, 16), (16, len(out["hyp"]))+1]
        prob_per_word = [np.mean(prob[word_slice]) for word_slice in word_slices]      # note: could use np.average with weights to speed up ?
    else:
        prob_per_word = np.mean(prob)
    return prob_per_word


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
        compute_confidence_scores: bool = True,
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
        self.compute_confidence_scores = compute_confidence_scores

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, *args):
        super().on_test_batch_end(trainer, pl_module, outputs, batch, *args)
        img_ids = pl_module.batch_id_fn(batch)
        hyps = self.decoder(outputs)["hyp"]
        probs = self.decoder(outputs)["prob-htr-char"]

        # compute mean confidence score and mean confidence score by word
        space_id = self.syms._sym2val[self.input_space]
        mean_probs = [np.mean(prob) for prob in probs]
        word_probs = [compute_word_prob(hyps[i], probs[i], space_id) for i in range(len(probs))]

        if len(word_probs[0]) == 3:
            print(word_probs[0])
            with open('col1.txt', 'a') as f:
                f.write(f"{word_probs[0][0]:.4f}\n")
            with open('col2.txt', 'a') as f:
                f.write(f"{word_probs[0][1]:.4f}\n")
            with open('col3.txt', 'a') as f:
                f.write(f"{word_probs[0][2]:.4f}\n")

        for i, (img_id, hyp, prob) in enumerate(zip(img_ids, hyps, mean_probs)):
            if self.use_symbols:
                hyp = [self.syms[v] for v in hyp]
                if self.convert_spaces:
                    hyp = [
                        self.output_space if sym == self.input_space else sym
                        for sym in hyp
                    ]
            if self.join_string is not None:
                hyp = self.join_string.join(str(x) for x in hyp)

            if self.compute_confidence_scores:
                self.write(
                    f"{img_id}{self.separator}{prob:.4f}{self.separator}{hyp}"
                    if self.include_img_ids
                    else f"{prob:.4f}{self.separator}{hyp}"
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
