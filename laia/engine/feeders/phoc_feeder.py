from laia.engine.feeders.feeder import Feeder
from laia.utils.symbols_table import SymbolsTable
from laia.utils.phoc import unigram_phoc, new_unigram_phoc

import torch


class PHOCFeeder(Feeder):
    def __init__(
        self, syms, levels, ignore_missing=True, new_phoc=False, parent_feeder=None
    ):
        super(PHOCFeeder, self).__init__(parent_feeder)
        assert isinstance(syms, (dict, SymbolsTable))
        assert isinstance(levels, (list, tuple))
        self._syms = syms
        self._levels = levels
        self._ignore_missing = ignore_missing
        self._phoc_func = new_unigram_phoc if new_phoc else unigram_phoc

    def _feed(self, x):
        assert isinstance(x, (list, tuple))
        return torch.tensor(
            [
                self._phoc_func(x, self._syms, self._levels, self._ignore_missing)
                for x in x
            ]
        )
