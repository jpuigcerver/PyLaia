import re

import pytest

from laia.callbacks import Decode
from laia.dummies import DummyEvaluator, DummyMNISTLines, DummyTrainer
from laia.losses.ctc_loss import transform_batch


class _TestDecoder:
    def __call__(self, batch_y_hat):
        _, xs = transform_batch(batch_y_hat)
        batch_size = len(xs)
        return {"hyp": [[1, 4, 11, 6, 0, 10] for _ in range(batch_size)]}


class __TestDecode(Decode):
    def __init__(self, img_id, hyp, *args, **kwargs):
        super().__init__(*args, decoder=_TestDecoder(), **kwargs)
        self.img_id = img_id
        self.hyp = re.escape(hyp)

    def write(self, value):
        assert re.match(self.img_id + self.hyp, value)


@pytest.mark.parametrize(
    ["kwargs", "img_id", "hyp"],
    [
        ({"include_img_ids": False}, "", "[1, 4, 11, 6, 0, 10]"),
        ({"use_symbols": True}, r"va-\d+ ", "['0', '3', '<space>', '5', '<ctc>', '9']"),
        (
            {"use_symbols": True, "convert_spaces": True},
            r"va-\d+ ",
            "['0', '3', ' ', '5', '<ctc>', '9']",
        ),
        ({"join_str": "-", "separator": " --- "}, r"va-\d+ --- ", "1-4-11-6-0-10"),
        ({"use_symbols": True, "join_str": ""}, r"va-\d+ ", "03<space>5<ctc>9"),
    ],
)
@pytest.mark.parametrize("num_processes", (1, 2))
def test_decode(tmpdir, num_processes, kwargs, img_id, hyp):
    module = DummyEvaluator()
    data_module = DummyMNISTLines(batch_size=2, va_n=12, samples_per_space=10)
    decode_callback = __TestDecode(img_id, hyp, syms=data_module.syms, **kwargs)
    trainer = DummyTrainer(
        default_root_dir=tmpdir,
        limit_test_batches=3,
        callbacks=[decode_callback],
        accelerator="ddp_cpu" if num_processes > 1 else None,
        num_processes=num_processes,
    )
    trainer.test(module, datamodule=data_module)
