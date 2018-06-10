from __future__ import absolute_import
from __future__ import print_function

import unittest

import torch
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence

try:
    from laia.decoders.ctc_lattice_generator import CTCLatticeGenerator
    import pywrapfst as fst

    class CTCLatticeGeneratorTest(unittest.TestCase):
        def setUp(self):
            self._x = torch.Tensor(
                [
                    [[1, 2, 3], [0, 0, 1]],
                    [[-1, 0, 2], [0, 3, 0]],
                    [[2, 0, 0], [3, 1, 2]],
                    [[-1, -2, -3], [-3, -2, -1]],
                ]
            )
            self._packed_x = pack_padded_sequence(
                input=Variable(self._x, requires_grad=False), lengths=[3, 2]
            )

        @staticmethod
        def _fst1():
            compiler = fst.Compiler()
            print("0 1 1 1 -1", file=compiler)
            print("0 1 2 2 -2", file=compiler)
            print("0 1 3 3 -3", file=compiler)
            print("1 2 1 1  1", file=compiler)
            print("1 2 2 2  0", file=compiler)
            print("1 2 3 3 -2", file=compiler)
            print("2 3 1 1 -2", file=compiler)
            print("2 3 2 2  0", file=compiler)
            print("2 3 3 3  0", file=compiler)
            print("3 4 1 1  1", file=compiler)
            print("3 4 2 2  2", file=compiler)
            print("3 4 3 3  3", file=compiler)
            print("4", file=compiler)
            f = compiler.compile()
            return f

        @staticmethod
        def _fst1_packed():
            compiler = fst.Compiler()
            print("0 1 1 1 -1", file=compiler)
            print("0 1 2 2 -2", file=compiler)
            print("0 1 3 3 -3", file=compiler)
            print("1 2 1 1  1", file=compiler)
            print("1 2 2 2  0", file=compiler)
            print("1 2 3 3 -2", file=compiler)
            print("2 3 1 1 -2", file=compiler)
            print("2 3 2 2  0", file=compiler)
            print("2 3 3 3  0", file=compiler)
            print("3", file=compiler)
            f = compiler.compile()
            return f

        @staticmethod
        def _fst2():
            compiler = fst.Compiler()
            print("0 1 1 1  0", file=compiler)
            print("0 1 2 2  0", file=compiler)
            print("0 1 3 3 -1", file=compiler)
            print("1 2 1 1  0", file=compiler)
            print("1 2 2 2 -3", file=compiler)
            print("1 2 3 3  0", file=compiler)
            print("2 3 1 1 -3", file=compiler)
            print("2 3 2 2 -1", file=compiler)
            print("2 3 3 3 -2", file=compiler)
            print("3 4 1 1  3", file=compiler)
            print("3 4 2 2  2", file=compiler)
            print("3 4 3 3  1", file=compiler)
            print("4", file=compiler)
            f = compiler.compile()
            return f

        @staticmethod
        def _fst2_packed():
            compiler = fst.Compiler()
            print("0 1 1 1  0", file=compiler)
            print("0 1 2 2  0", file=compiler)
            print("0 1 3 3 -1", file=compiler)
            print("1 2 1 1  0", file=compiler)
            print("1 2 2 2 -3", file=compiler)
            print("1 2 3 3  0", file=compiler)
            print("2", file=compiler)
            f = compiler.compile()
            return f

        @staticmethod
        def normalize_fst(f):
            f2 = f.copy()
            z = fst.shortestdistance(fst.arcmap(f, map_type="to_log"), reverse=True)[0]
            for s in f2.states():
                w = f2.final(s)
                nw = fst.Weight(f2.weight_type(), float(w) - float(z))
                f2.set_final(s, nw)
            return f2

        def tensor_test(self):
            latgen = CTCLatticeGenerator()
            lats = latgen(self._x)
            self.assertTrue(fst.equivalent(lats[0], self._fst1()))
            self.assertTrue(fst.equivalent(lats[1], self._fst2()))

        def tensor_normalize_test(self):
            latgen = CTCLatticeGenerator(normalize=True)
            lats = latgen(self._x)
            self.assertTrue(fst.equivalent(lats[0], self.normalize_fst(self._fst1())))
            self.assertTrue(fst.equivalent(lats[1], self.normalize_fst(self._fst2())))

        def packed_tensor_normalize_test(self):
            latgen = CTCLatticeGenerator(normalize=True)
            lats = latgen(self._packed_x)
            self.assertTrue(
                fst.equivalent(lats[0], self.normalize_fst(self._fst1_packed()))
            )
            self.assertTrue(
                fst.equivalent(lats[1], self.normalize_fst(self._fst2_packed()))
            )


except ImportError:
    import warnings

    warnings.warn("Module pywrapfst is not installed and tests cannot run")
    fst = None

if __name__ == "__main__":
    unittest.main()
