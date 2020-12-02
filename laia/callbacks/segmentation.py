import sys
from typing import Callable, List, Optional, Tuple, Union

import pytorch_lightning as pl
from tqdm.auto import tqdm

from laia.decoders import CTCGreedyDecoder
from laia.utils import SymbolsTable


class Segmentation(pl.Callback):
    def __init__(
        self,
        syms: Union[dict, SymbolsTable],
        decoder: Optional[Callable] = CTCGreedyDecoder(),
        segmentation: str = "char",
        input_space: str = "<space>",
        separator: str = " ",
        include_img_ids: bool = True,
    ):
        super().__init__()
        self.syms = syms
        self.decoder = decoder
        self.segmentation = segmentation
        self.input_space = input_space
        self.separator = separator
        self.include_img_ids = include_img_ids

    @staticmethod
    def char(
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

    @staticmethod
    def word(
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

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, *args, **kwargs):
        super().on_test_batch_end(trainer, pl_module, outputs, batch, *args, **kwargs)
        batch_x = pl_module.batch_input_fn(batch)
        img_ids = pl_module.batch_id_fn(batch)
        decoding = self.decoder(outputs, segmentation=True)
        for i, img_id in enumerate(img_ids):
            out = [self.syms[v] for v in decoding["hyp"][i]]
            h, w = batch_x.sizes[i, -2:].tolist()
            out = self.char(out, decoding["segmentation"][i], h, width=w)
            if self.segmentation == "word":
                out = self.word(out, self.input_space)
            self.write(
                f"{img_id}{self.separator}{out}" if self.include_img_ids else str(out),
            )

    def write(self, value):
        return tqdm.write(value, file=sys.stdout)
