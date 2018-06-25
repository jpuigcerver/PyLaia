from __future__ import absolute_import

import torch
import string
from torch.autograd import Variable
from laia.data.padding_collater import PaddedTensor


def generate_backprop_test(
    tensor_type,
    device_func,
    module,
    module_kwargs,
    batch_data,
    batch_sizes,
    cost_function,
    padded_cost_function,
):
    def run(self):
        m = module(**module_kwargs)
        m = device_func(m.type(tensor_type))
        m.train()

        # Convert batch input and batch sizes to appropriate type
        x = device_func(batch_data.type(tensor_type))
        xs = device_func(torch.LongTensor(batch_sizes))

        # Check model for normal tensor inputs
        m.zero_grad()
        cost = cost_function(m(Variable(x)))
        cost.backward()
        for n, p in m.named_parameters():
            self.assertIsNotNone(
                p.grad, msg="Parameter {!r} does not have a gradient".format(n)
            )
            sp = torch.abs(p.grad.data).sum()
            self.assertNotAlmostEqual(
                sp,
                0.0,
                msg="Gradients for parameter {!r} are close to 0 (:g)".format(n, sp),
            )

        # Check model for padded tensor inputs
        m.zero_grad()
        cost = padded_cost_function(
            m(PaddedTensor(data=Variable(x), sizes=Variable(xs)))
        )
        cost.backward()
        for n, p in m.named_parameters():
            self.assertIsNotNone(
                p.grad, msg="Parameter {!r} does not have a gradient".format(n)
            )
            sp = torch.abs(p.grad.data).sum()
            self.assertNotAlmostEqual(
                sp,
                0.0,
                msg="Gradients for parameter {!r} are close to 0 (:g)".format(n, sp),
            )

    return run


def generate_backprop_tests(cls, tensor_types, tests):
    devices = [("cpu", lambda x: x.cpu())]
    if torch.cuda.is_available():
        devices += [("gpu", lambda x: x.cuda())]

    for tn, tt in tensor_types:
        for dn, df in devices:
            for name_pattern, generator_kwargs in tests:
                fields = [
                    t[1]
                    for t in string.Formatter().parse(name_pattern)
                    if t[1] is not None
                ]
                assert fields == ["", ""]
                test_name = ("test_" + name_pattern).format(tn, dn)
                setattr(
                    cls,
                    test_name,
                    generate_backprop_test(
                        tensor_type=tt, device_func=df, **generator_kwargs
                    ),
                )


def generate_backprop_floating_point_tests(cls, tests):
    generate_backprop_tests(
        cls, [("f32", "torch.FloatTensor"), ("f64", "torch.DoubleTensor")], tests
    )
