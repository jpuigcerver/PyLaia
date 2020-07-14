from typing import Union, Dict, List

from laia.utils.symbols_table import SymbolsTable


class ToTensor:
    def __init__(self, syms: Union[Dict, SymbolsTable]) -> None:
        assert isinstance(syms, (Dict, SymbolsTable))
        self._syms = syms

    def __call__(self, x: str) -> List[int]:
        return [
            self._syms.get(c, None) if isinstance(self._syms, Dict) else self._syms[c]
            for c in x.split()
        ]

    def __repr__(self) -> str:
        return "text.{}()".format(self.__class__.__name__)
