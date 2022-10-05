from pathlib import Path
from typing import Iterator, Optional, Tuple, Union


class SymbolsTable:
    """
    Bijective mapping between symbols (strings) and integers

    Args:
        filepath: Filepath to the symbols file
    """

    def __init__(
        self,
        filepath: Optional[Union[str, Path]] = None,
        from_dict: Optional[dict] = None,
    ):
        self._sym2val, self._val2sym = dict(), dict()
        if filepath:
            self.load(filepath)
        elif from_dict:
            self.load_dict(from_dict)

    def clear(self):
        self._sym2val, self._val2sym = dict(), dict()

    def load(self, f: Union[str, Path]):
        f = open(f)
        self.clear()
        try:
            lines = [line.split() for line in f if len(line.split())]
            for s, v in lines:
                self.add(s, int(v))
        except Exception:
            raise
        finally:
            f.close()

    def load_dict(self, d: dict):
        self.clear()
        for index, symbol in d.items():
            self.add(symbol, int(index))

    def save(self, f: Union[str, Path]):
        f = open(f, "w")
        max_len = max(len(s) for s in self._sym2val)
        for v, s in self._val2sym.items():
            f.write(f"{s:>{max_len}} {v}\n")
        f.close()

    def __len__(self) -> int:
        return len(self._val2sym)

    def __getitem__(self, x: Union[str, int]) -> Optional[Union[str, int]]:
        if isinstance(x, int):
            return self._val2sym.get(x, None)
        if isinstance(x, str):
            return self._sym2val.get(x, None)
        raise ValueError("SymbolsTable contains pairs of integers and strings")

    def __iter__(self) -> Iterator[Tuple[str, int]]:
        for v, s in self._val2sym.items():
            yield s, v

    def __contains__(self, x: Union[str, int]) -> bool:
        if isinstance(x, int):
            return x in self._val2sym
        if isinstance(x, str):
            return x in self._sym2val
        raise ValueError(
            f'SymbolsTable contains pairs of integers and strings, found "{x}"'
        )

    def add(self, symbol: str, value: int):
        if not isinstance(symbol, str):
            raise KeyError(
                f"Symbol must be a string, but type {type(symbol)} was given"
            )
        if not isinstance(value, int):
            raise KeyError(
                f"Value must be an integer, but type {type(value)} was given"
            )
        old_val = self._sym2val.get(symbol, None)
        old_sym = self._val2sym.get(value, None)
        if old_val is None and old_sym is None:
            self._sym2val[symbol] = value
            self._val2sym[value] = symbol
        elif old_val == value and old_sym == symbol:
            # Nothing changes, so just ignore the add() operation
            return
        elif old_val is not None:
            raise KeyError(
                f'Symbol "{symbol}" was already present '
                f'in the table (assigned to value "{old_val}")'
            )
        elif old_sym is not None:
            raise KeyError(
                f'Value "{value}" was already present '
                f'in the table (assigned to symbol "{old_sym}")'
            )
