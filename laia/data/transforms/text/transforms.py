from typing import Dict, List, Union

import laia.common.logging as log
from laia.utils.symbols_table import SymbolsTable

_logger = log.get_logger(__name__)


class ToTensor:
    def __init__(self, syms: Union[Dict, SymbolsTable]) -> None:
        assert isinstance(syms, (Dict, SymbolsTable))
        self._syms = syms

    def __call__(self, x: str) -> List[int]:
        values = []
        for c in x.split():
            v = (
                self._syms.get(c, None)
                if isinstance(self._syms, Dict)
                else self._syms[c]
            )
            if v is None:
                _logger.error('Could not find "{}" in the symbols table', c)
            values.append(v)
        return values

    def __repr__(self) -> str:
        return f"text.{self.__class__.__name__}()"
