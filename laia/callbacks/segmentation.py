import sys
from typing import List, Optional, Tuple, Union

import pytorch_lightning as pl
from tqdm.auto import tqdm

from laia.utils import SymbolsTable


def char_segmentation(
    txt: List[str], seg: List[int], height: int, width: Optional[int] = None
) -> List[Tuple[str, int, int, int, int]]:
    assert any(len(txt) + i == len(seg) for i in (1, 2))
    if width:
        # Scale the width
        max_pos = seg[-1]
        assert max_pos <= width
        seg = [(x * width // max_pos) for x in seg]
    # Convert (0-based start, non-inclusive end)
    # to (1-based start, inclusive end)
    seg[0] += 1
    seg[-1] += 1
    return [
        # (value, p1=(x, y), p2=(x, y))
        (txt[j], seg[j], 1, seg[j + 1] - 1, height)
        for j in range(len(txt))
    ]


def word_segmentation(
    segmentation: List[Tuple[str, int, int, int, int]],
    space: str,
    include_spaces: bool = True,
) -> List[Tuple[str, int, int, int, int]]:
    pairs = list(zip(segmentation, segmentation[1:]))
    assert all(s1[3] + 1 == s2[1] for s1, s2 in pairs)
    assert all(s1[2] == s2[2] for s1, s2 in pairs)
    assert all(s1[4] == s2[4] for s1, s2 in pairs)
    out = []
    w, w_x1, w_x2 = "", None, None
    for i, (c, x1, y1, x2, y2) in enumerate(segmentation):
        if c == space:
            if w:
                out.append((w, w_x1, y1, w_x2, y2))
            if include_spaces:
                out.append((c, x1, y1, x2, y2))
            w = ""
            w_x1 = x2 + 1
        else:
            if i == 0:
                w_x1 = x1
            w += c
            w_x2 = x2
        if i == len(segmentation) - 1 and w:
            out.append((w, w_x1, y1, x2, y2))
    return out


class Segmentation(pl.callbacks.Callback):
    def __init__(
        self,
        syms: Union[dict, SymbolsTable],
        segmentation: str = "char",
        input_space: str = "<space>",
        join_str: Optional[str] = None,
        separator: str = " ",
        print_img_ids: bool = True,
    ):
        super().__init__()
        self.syms = syms
        self.segmentation = segmentation
        self.input_space = input_space
        self.join_str = join_str
        self.separator = separator
        self.print_img_ids = print_img_ids

    def on_test_batch_end(self, trainer, pl_module, batch, *args):
        super().on_test_batch_end(trainer, pl_module, batch, *args)
        batch_x = pl_module.batch_input_fn(batch)
        img_ids = pl_module.batch_id_fn(batch)
        decoder = pl_module.decoder
        for i, (img_id, out) in enumerate(zip(img_ids, decoder.output)):
            out = [self.syms[v] for v in out]
            h, w = batch_x.sizes[i].tolist()
            out = char_segmentation(out, decoder.segmentation[i], h, width=w)
            if self.segmentation == "word":
                out = word_segmentation(out, self.input_space)
            tqdm.write(
                f"{img_id}{self.separator}{out}" if self.print_img_ids else out,
                file=sys.stdout,
            )
