import os
import unittest
from io import StringIO
from tempfile import NamedTemporaryFile

from laia.utils.symbols_table import SymbolsTable


class SymbolsTableTest(unittest.TestCase):
    def test_empty(self):
        st = SymbolsTable()
        self.assertEqual(len(st), 0)

    def test_add(self):
        st = SymbolsTable()
        st.add("<eps>", 0)
        self.assertEqual(len(st), 1)
        st.add("a", 1)
        self.assertEqual(len(st), 2)

    def test_add_valid_repeated(self):
        st = SymbolsTable()
        st.add("<eps>", 0)
        st.add("<eps>", 0)
        self.assertEqual(len(st), 1)

    def test_add_repeated_symbol(self):
        st = SymbolsTable()
        st.add("a", 0)
        with self.assertRaises(
            KeyError,
            msg='Symbol "a" was already present in the table (assigned to value "0")',
        ):
            st.add("a", 1)

    def test_add_repeated_value(self):
        st = SymbolsTable()
        st.add("a", 0)
        with self.assertRaises(
            KeyError,
            msg='Value "0" was already present in the table (assigned to symbol "a")',
        ):
            st.add("b", 0)

    def test_getitem(self):
        st = SymbolsTable()
        st.add("a", 1)
        st.add("b", 2)
        self.assertEqual(st["a"], 1)
        self.assertEqual(st["b"], 2)
        self.assertEqual(st[1], "a")
        self.assertEqual(st[2], "b")
        self.assertEqual(st[-9], None)
        self.assertEqual(st["c"], None)

    def test_iterator(self):
        st = SymbolsTable()
        st.add("a", 1)
        st.add("b", 2)
        it = iter(st)
        self.assertEqual(next(it), ("a", 1))
        self.assertEqual(next(it), ("b", 2))
        with self.assertRaises(StopIteration):
            next(it)

    def test_load(self):
        table_file = StringIO("\n\na   1\nb     2\n")
        st = SymbolsTable(table_file)
        self.assertEqual(len(st), 2)
        self.assertEqual(st["a"], 1)
        self.assertEqual(st["b"], 2)
        self.assertEqual(st[1], "a")
        self.assertEqual(st[2], "b")

    def test_load_value_error(self):
        table_file = StringIO("\n\na   1\nb     c\n")
        with self.assertRaises(ValueError):
            SymbolsTable(table_file)

    def test_save(self):
        st = SymbolsTable()
        st.add("a", 1)
        st.add("b", 2)
        table_file = NamedTemporaryFile(delete=False)
        st.save(table_file)
        with open(table_file.name) as f:
            table_content = f.read()
        self.assertEqual(table_content, "a 1\nb 2\n")
        os.remove(table_file.name)


if __name__ == "__main__":
    unittest.main()
