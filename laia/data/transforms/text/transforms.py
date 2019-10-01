from typing import Union, Dict

from laia.utils.symbols_table import SymbolsTable


class ToTensor:
    def __init__(self, syms: Union[Dict, SymbolsTable]) -> None:
        assert isinstance(syms, (Dict, SymbolsTable))
        self._syms = syms

    def __call__(self, x):
        return [self._syms[c] for c in x]

    def __repr__(self) -> str:
        return "text.{}()".format(self.__class__.__name__)
