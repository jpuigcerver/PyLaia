from laia.utils.symbols_table import SymbolsTable


class TextToTensor(object):

    def __init__(self, syms):
        assert isinstance(syms, (dict, SymbolsTable))
        self._syms = syms

    def __call__(self, x):
        x = [self._syms[c] for c in x]
        return x
