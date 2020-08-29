import torch

from laia.data.padding_collater import PaddedTensor


def generate_backprop_test(
    dtype,
    device,
    module,
    module_kwargs,
    batch_data,
    batch_sizes,
    cost_function,
    padded_cost_function,
):
    def run(self):
        m = module(**module_kwargs).to(device, dtype=dtype).train()
        # Convert batch input and batch sizes to appropriate type
        x = batch_data.to(device, dtype=dtype)
        xs = torch.tensor(batch_sizes, device=device)

        # Check model for normal tensor inputs
        m.zero_grad()
        cost = cost_function(m(x))
        cost.backward()
        for n, p in m.named_parameters():
            self.assertIsNotNone(p.grad, msg=f"Parameter {n} does not have a gradient")
            sp = torch.abs(p.grad).sum().item()
            self.assertNotAlmostEqual(
                sp, 0.0, msg=f"Gradients for parameter {n} are close to 0 ({sp:g})"
            )

        # Check model for padded tensor inputs
        m.zero_grad()
        cost = padded_cost_function(m(PaddedTensor(x, xs)))
        cost.backward()
        for n, p in m.named_parameters():
            self.assertIsNotNone(p.grad, msg=f"Parameter {n} does not have a gradient")
            sp = torch.abs(p.grad).sum().item()
            self.assertNotAlmostEqual(
                sp, 0.0, msg=f"Gradients for parameter {n} are close to 0 ({sp:g})"
            )

    return run


def generate_backprop_tests(cls, dtypes, tests):
    devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
    for dtype in dtypes:
        for device in devices:
            for name_pattern, kwargs in tests:
                setattr(
                    cls,
                    ("test_" + name_pattern).format(str(dtype)[6:], device),
                    generate_backprop_test(dtype=dtype, device=device, **kwargs),
                )


def generate_backprop_floating_point_tests(cls, tests):
    generate_backprop_tests(cls, (torch.float, torch.double), tests)
