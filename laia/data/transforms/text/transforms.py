from laia.utils.symbols_table import SymbolsTable


class ToTensor(object):
    def __init__(self, syms):
        assert isinstance(syms, (dict, SymbolsTable))
        self._syms = syms

    def __call__(self, x):
        return [self._syms[c] for c in x]

    def __repr__(self):
        return "text.{}()".format(self.__class__.__name__)
