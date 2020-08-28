import unittest

from laia.data.transforms.text import ToTensor
from laia.utils import SymbolsTable


class TransformerTest(unittest.TestCase):
    def test_call_with_dict(self):
        t = ToTensor({"a": 0, "b": 1, "<space>": 2, "<": 3})
        x = "a < b <space> a <sp"
        y = t(x)
        actual = [0, 3, 1, 2, 0, None]
        self.assertEqual(actual, y)

    def test_call_with_symbols_table(self):
        st = SymbolsTable()
        for k, v in {"a": 0, "b": 1, "<space>": 2, "<": 3}.items():
            st.add(k, v)
        t = ToTensor(st)
        x = "a < b <space> a <sp"
        y = t(x)
        actual = [0, 3, 1, 2, 0, None]
        self.assertEqual(actual, y)
