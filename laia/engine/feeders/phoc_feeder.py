from laia.engine.feeders.feeder import Feeder
from laia.utils.symbols_table import SymbolsTable
from laia.utils.phoc import unigram_phoc

import torch


class PHOCFeeder(Feeder):
    def __init__(self, syms, levels, parent_feeder=None):
        super(PHOCFeeder, self).__init__(parent_feeder)
        assert isinstance(syms, (dict, SymbolsTable))
        assert isinstance(levels, (list, tuple))
        self._syms = syms
        self._levels = levels

    def _feed(self, batch):
        assert isinstance(batch, (list, tuple))
        return torch.Tensor([unigram_phoc(x, self._syms, self._levels)
                             for x in batch])
