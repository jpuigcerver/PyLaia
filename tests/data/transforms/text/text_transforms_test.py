from laia.data.transforms.text import ToTensor
from laia.utils import SymbolsTable


def test_call_with_dict(caplog):
    t = ToTensor({"a": 0, "b": 1, "<space>": 2, "<": 3})
    x = "a < b <space> a <sp"
    y = t(x)
    assert y == [0, 3, 1, 2, 0, None]
    assert caplog.messages.count('Could not find "<sp" in the symbols table') == 1


def test_call_with_symbols_table(caplog):
    st = SymbolsTable()
    for k, v in {"a": 0, "b": 1, "<space>": 2, "<": 3}.items():
        st.add(k, v)
    t = ToTensor(st)
    x = "a < b <space> a ö"
    y = t(x)
    assert y == [0, 3, 1, 2, 0, None]
    assert caplog.messages.count('Could not find "ö" in the symbols table') == 1
