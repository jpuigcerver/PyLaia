from builtins import str


class SymbolsTable(object):
    def __init__(self, f=None):
        self._sym2val, self._val2sym = dict(), dict()
        if f:
            self.load(f)

    def clear(self):
        self._sym2val, self._val2sym = dict(), dict()

    def load(self, f):
        if isinstance(f, str):
            f = open(f, 'r')
        self.clear()
        try:
            for n, line in enumerate(f, 1):
                line = line.split()
                if len(line) == 0: continue
                s, v = line[0], int(line[1])
                self.add(s, v)
        except Exception:
            raise
        finally:
            f.close()

    def save(self, f):
        if isinstance(f, str):
            f = open(f, 'w')
        max_len = max([len(s) for s in self._sym2val])
        for v, s in self._val2sym.items():
            f.write('%*s %d\n' % (max_len, s, v))
        f.close()

    def __len__(self):
        return len(self._val2sym)

    def __getitem__(self, x):
        if isinstance(x, int):
            return self._val2sym.get(x, None)
        elif isinstance(x, str):
            return self._sym2val.get(x, None)
        else:
            return None

    def __iter__(self):
        for v, s in self._val2sym.items():
            yield s, v

    def add(self, symbol, value):
        if not isinstance(symbol, str):
            raise KeyError(
                'Symbol must be a string, but type %s was given' % type(symbol))
        if not isinstance(value, int):
            raise KeyError(
                'Value must be an integer, but type %s was given' % type(value))

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
                ('Symbol "%s" was already present in the table ('
                 'assigned to value %d)') % (symbol, old_val))
        elif old_sym is not None:
            raise KeyError(
                ('Value "%d" was already present in the table ('
                 'assigned to symbol "%s"') % (value, old_sym))
