from __future__ import absolute_import

import io
from torch._six import string_classes


class SymbolsTable(object):
    def __init__(self, f=None):
        self._sym2val, self._val2sym = dict(), dict()
        if f:
            self.load(f)

    def clear(self):
        self._sym2val, self._val2sym = dict(), dict()

    def load(self, f, encoding="utf8"):
        if isinstance(f, string_classes):
            f = io.open(f, "r", encoding=encoding)
        self.clear()
        try:
            lines = [line.split() for line in f if len(line.split())]
            for s, v in lines:
                self.add(s, int(v))
        except Exception:
            raise
        finally:
            f.close()

    def save(self, f, encoding="utf8"):
        if isinstance(f, string_classes):
            f = io.open(f, "w", encoding=encoding)
        max_len = max(len(s) for s in self._sym2val)
        for v, s in self._val2sym.items():
            f.write("{:>{w}} {}\n".format(s, v, w=max_len).encode(encoding))
        f.close()

    def __len__(self):
        return len(self._val2sym)

    def __getitem__(self, x):
        if isinstance(x, int):
            return self._val2sym.get(x, None)
        elif isinstance(x, string_classes):
            return self._sym2val.get(x, None)
        else:
            raise ValueError("SymbolsTable contains pairs of integers and strings")

    def __iter__(self):
        for v, s in self._val2sym.items():
            yield s, v

    def __contains__(self, x):
        if isinstance(x, int):
            return x in self._val2sym
        elif isinstance(x, string_classes):
            return x in self._sym2val
        else:
            raise ValueError("SymbolsTable contains pairs of integers and strings")

    def add(self, symbol, value):
        if not isinstance(symbol, string_classes):
            raise KeyError(
                "Symbol must be a string, " "but type {} was given".format(type(symbol))
            )
        if not isinstance(value, int):
            raise KeyError(
                "Value must be an integer, " "but type {} was given".format(type(value))
            )

        old_val = self._sym2val.get(symbol, None)
        old_sym = self._val2sym.get(value, None)
        if old_val is None and old_sym is None:
            self._sym2val[symbol] = value
            self._val2sym[value] = symbol
        elif old_val == value and old_sym == symbol:
            # Nothing changes, so just ignore the add() operation
            pass
        elif old_val is not None:
            raise KeyError(
                'Symbol "{}" was already present in the table '
                "(assigned to value {})".format(symbol, old_val)
            )
        elif old_sym is not None:
            raise KeyError(
                'Value "{}" was already present in the table '
                '(assigned to symbol "{}")'.format(value, old_sym)
            )
