import unittest

from laia.engine.engine_exception import EngineException


class EngineExceptionTest(unittest.TestCase):
    def test_raise(self):
        def f():
            raise EngineException(
                epoch=15,
                iteration=20,
                batch={"ids": [1, 2], "data": [[1, 2, 3], [4, 5, 6]]},
            )

        self.assertRaises(EngineException, f)

    def test_raise_from(self):
        def f1():
            raise TypeError("Original Exception")

        def f2():
            try:
                f1()
            except TypeError as exc:
                raise EngineException(
                    epoch=1, iteration=2, batch="Batch", cause=exc
                ) from exc

        self.assertRaisesRegex(EngineException, r'^Exception "TypeError\(.*$', f2)


if __name__ == "__main__":
    unittest.main()
