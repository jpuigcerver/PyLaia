import re

import pytest

from laia.callbacks import Netout
from laia.dummies import DummyEvaluator, DummyMNISTLines, DummyTrainer
from laia.utils import ArchiveMatrixWriter


class __TestWriter(ArchiveMatrixWriter):
    def write(self, key, matrix):
        assert re.match(r"va-\d+", key)
        assert matrix.size() == (3, 10)


def identity(x):
    return x


@pytest.mark.parametrize("num_processes", (1, 2))
def test_segmentation_callback(tmpdir, num_processes):
    data_module = DummyMNISTLines(batch_size=2, va_n=12)
    trainer = DummyTrainer(
        default_root_dir=tmpdir,
        limit_test_batches=3,
        callbacks=[Netout(writers=[__TestWriter(None)], output_transform=identity)],
        accelerator="ddp_cpu" if num_processes > 1 else None,
        num_processes=num_processes,
    )
    trainer.test(DummyEvaluator(), datamodule=data_module)
