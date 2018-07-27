import unittest

import numpy as np
import torch
from torch.autograd import Variable
from torch.nn.functional import adaptive_avg_pool2d
from torch.nn.functional import adaptive_max_pool2d

from laia.data import PaddedTensor
from laia.nn.image_pooling_sequencer import ImagePoolingSequencer

try:
    import nnutils_pytorch

    nnutils_installed = True
except ImportError:
    nnutils_installed = False


class ImagePoolingSequencerTest(unittest.TestCase):
    def test_bad_sequencer(self):
        self.assertRaises(ValueError, ImagePoolingSequencer, sequencer="")
        self.assertRaises(ValueError, ImagePoolingSequencer, sequencer="foo")
        self.assertRaises(ValueError, ImagePoolingSequencer, sequencer="maxpool")
        self.assertRaises(ValueError, ImagePoolingSequencer, sequencer="avgpool-")
        self.assertRaises(ValueError, ImagePoolingSequencer, sequencer="maxpool-c")


def _generate_test(sequencer, poolsize, columnwise, x, output_size):
    def _test(self):
        m = ImagePoolingSequencer(
            sequencer="{}-{}".format(sequencer, poolsize), columnwise=columnwise
        )
        self.assertListEqual(list(m(x).size()), output_size)

    return _test


def _generate_failing_test(sequencer, poolsize, columnwise, x):
    def _test(self):
        m = ImagePoolingSequencer(
            sequencer="{}-{}".format(sequencer, poolsize), columnwise=columnwise
        )

        def wrap_call():
            return m(x)

        self.assertRaises(ValueError, wrap_call)

    return _test


def _generate_gradcheck_test(
    sequencer, ref_func, poolsize, columnwise, use_cuda, x, xs
):
    def _test(self):
        m = ImagePoolingSequencer(
            sequencer="{}-{}".format(sequencer, poolsize), columnwise=columnwise
        )
        xv = Variable(x, requires_grad=True)
        if use_cuda:
            m = m.cuda()
            xv = xv.cuda()
            xsv = Variable(torch.cuda.LongTensor(xs))
        else:
            m = m.cpu()
            xv = xv.cpu()
            xsv = Variable(torch.LongTensor(xs))

        yv = m(PaddedTensor(data=xv, sizes=xsv))
        dx1, = torch.autograd.grad(yv.data.sum(), (xv,))

        for i, (xk, xsk) in enumerate(zip(x, xs)):
            xk = xk[:, : xsk[0], : xsk[1]].unsqueeze(0)
            xk = Variable(xk.cuda() if use_cuda else xk.cpu(), requires_grad=True)
            if columnwise:
                yk = ref_func(xk, output_size=(poolsize, xsk[1]))
            else:
                yk = ref_func(xk, output_size=(xsk[0], poolsize))
            dxk, = torch.autograd.grad(yk.sum(), (xk,))
            np.testing.assert_almost_equal(
                actual=dx1.data[i, :, : xsk[0], : xsk[1]].cpu().numpy(),
                desired=dxk.data[0].cpu().numpy(),
                err_msg="Failed {} with sample {}".format(sequencer, i),
            )

    return _test


devices = [("cpu", False)]
if torch.cuda.is_available():
    devices += [("gpu", True)]

tensor_types = [("f32", "torch.FloatTensor"), ("f64", "torch.DoubleTensor")]

if nnutils_installed:
    for sequencer, ref_func in [
        ("avgpool", adaptive_avg_pool2d),
        ("maxpool", adaptive_max_pool2d),
    ]:
        for type_name, tensor_type in tensor_types:
            for device_name, use_cuda in devices:
                setattr(
                    ImagePoolingSequencerTest,
                    "test_grad_{}_{}_{}".format(sequencer, type_name, device_name),
                    _generate_gradcheck_test(
                        sequencer=sequencer,
                        ref_func=ref_func,
                        poolsize=10,
                        columnwise=True,
                        use_cuda=use_cuda,
                        x=torch.randn(3, 4, 17, 19).type(tensor_type),
                        xs=[[17, 19], [11, 13], [13, 11]],
                    ),
                )

for sequencer in ["none", "maxpool", "avgpool"] if nnutils_installed else ["none"]:
    setattr(
        ImagePoolingSequencerTest,
        "test_tensor_{}_col".format(sequencer),
        _generate_test(
            sequencer=sequencer,
            poolsize=10,
            columnwise=True,
            x=torch.randn(2, 3, 10, 11),
            output_size=[11, 2, 3 * 10],
        ),
    )
    setattr(
        ImagePoolingSequencerTest,
        "test_tensor_{}_row".format(sequencer),
        _generate_test(
            sequencer=sequencer,
            poolsize=11,
            columnwise=False,
            x=torch.randn(2, 3, 10, 11),
            output_size=[10, 2, 3 * 11],
        ),
    )

for columnwise in [True, False]:
    setattr(
        ImagePoolingSequencerTest,
        "test_tensor_bad_input_{}".format("col" if columnwise else "row"),
        _generate_failing_test(
            sequencer="none",
            poolsize=9,
            columnwise=columnwise,
            x=torch.randn(2, 3, 4, 5),
        ),
    )

if __name__ == "__main__":
    unittest.main()
