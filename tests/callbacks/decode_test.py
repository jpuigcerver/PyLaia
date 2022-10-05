import pytest
import regex as re

from laia.callbacks import Decode
from laia.dummies import DummyEvaluator, DummyMNISTLines, DummyTrainer
from laia.losses.ctc_loss import transform_batch


class _TestDecoder:
    def __call__(self, batch_y_hat):
        _, xs = transform_batch(batch_y_hat)
        batch_size = len(xs)
        return {
            "hyp": [[1, 4, 11, 6, 0, 10] for _ in range(batch_size)],
            "prob-htr-char": [
                [0.9, 0.9, 0.9, 0.9, 0.9, 0.9] for _ in range(batch_size)
            ],
        }


class __TestDecode(Decode):
    def __init__(self, img_id, hyp, prob, *args, **kwargs):
        super().__init__(*args, decoder=_TestDecoder(), **kwargs)
        self.img_id = img_id
        self.hyp = re.escape(hyp)
        self.prob = prob

    def write(self, value):
        sep_1 = re.escape(self.separator if self.img_id else "")
        sep_2 = re.escape(self.separator if self.prob else "")
        output = f"{self.img_id}{sep_1}{self.prob}{sep_2}{self.hyp}"
        assert re.match(output, value)


@pytest.mark.parametrize(
    ["kwargs", "img_id", "prob", "hyp"],
    [
        ({"include_img_ids": False}, "", "", "[1, 4, 11, 6, 0, 10]"),
        (
            {"include_img_ids": False, "print_line_confidence_scores": True},
            "",
            r"0\.\d\d",
            "[1, 4, 11, 6, 0, 10]",
        ),
        (
            {"include_img_ids": False, "print_word_confidence_scores": True},
            "",
            r"\['0\.\d\d', '0\.\d\d'\]",
            "[1, 4, 11, 6, 0, 10]",
        ),
        (
            {
                "include_img_ids": False,
                "print_line_confidence_scores": True,
                "separator": "|",
            },
            "",
            r"0\.\d\d",
            "[1, 4, 11, 6, 0, 10]",
        ),
        (
            {
                "include_img_ids": False,
                "print_word_confidence_scores": True,
                "separator": "|",
            },
            "",
            r"\['0\.\d\d', '0\.\d\d'\]",
            "[1, 4, 11, 6, 0, 10]",
        ),
        ({"include_img_ids": False}, "", "", "[1, 4, 11, 6, 0, 10]"),
        (
            {"use_symbols": True, "print_line_confidence_scores": True},
            r"va-\d+",
            r"0\.\d\d",
            "['0', '3', '<space>', '5', '<ctc>', '9']",
        ),
        (
            {
                "use_symbols": True,
                "convert_spaces": True,
                "print_line_confidence_scores": True,
            },
            r"va-\d+",
            r"0\.\d\d",
            "['0', '3', ' ', '5', '<ctc>', '9']",
        ),
        (
            {
                "join_string": "-",
                "print_line_confidence_scores": True,
                "separator": " --- ",
            },
            r"va-\d+",
            r"0\.\d\d",
            "1-4-11-6-0-10",
        ),
        (
            {
                "use_symbols": True,
                "join_string": "",
                "print_line_confidence_scores": False,
            },
            r"va-\d+",
            "",
            "03<space>5<ctc>9",
        ),
        pytest.param(
            {},
            "tr",
            "",
            "",
            marks=pytest.mark.xfail(reason="Check write called", strict=True),
        ),
    ],
)
@pytest.mark.parametrize("num_processes", (1, 2))
def test_decode(tmpdir, num_processes, kwargs, img_id, hyp, prob):
    module = DummyEvaluator()
    data_module = DummyMNISTLines(batch_size=2, va_n=12, samples_per_space=10)
    decode_callback = __TestDecode(img_id, hyp, prob, syms=data_module.syms, **kwargs)
    trainer = DummyTrainer(
        default_root_dir=tmpdir,
        limit_test_batches=3,
        callbacks=[decode_callback],
        accelerator="ddp_cpu" if num_processes > 1 else None,
        num_processes=num_processes,
    )
    trainer.test(module, datamodule=data_module)
