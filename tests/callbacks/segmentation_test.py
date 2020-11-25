import re

import pytest

from laia.callbacks.segmentation import Segmentation
from laia.dummies import DummyEvaluator, DummyMNISTLines, DummyTrainer


def test_char_segmentation_empty():
    with pytest.raises(AssertionError):
        Segmentation.char([""], [], 1)


def test_char_segmentation():
    txt = ["a", "b", "c"]
    seg = [0, 3, 5, 7, 10]
    x = Segmentation.char(txt, seg, 1)
    e = [("a", 1, 1, 2, 1), ("b", 3, 1, 4, 1), ("c", 5, 1, 6, 1)]
    assert x == e


def test_char_segmentation_scaling():
    txt = ["a", "b", "c"]
    seg = [0, 3, 5, 7, 10]
    x = Segmentation.char(txt, seg, 1, width=100)
    e = [("a", 1, 1, 29, 1), ("b", 30, 1, 49, 1), ("c", 50, 1, 69, 1)]
    assert x == e


def test_char_segmentation_scaling_error():
    with pytest.raises(AssertionError):
        Segmentation.char(["a"], [0, 1, 100], 1, width=50)


def test_word_segmentation_empty():
    x = Segmentation.word([], " ", include_spaces=True)
    assert x == []


@pytest.mark.parametrize(
    "s",
    [
        [("a", 1, 1, 3, 10), ("b", 4, 5, 10, 10)],  # different y1
        [("a", 1, 1, 3, 10), ("b", 4, 1, 10, 5)],  # different y2
        [("a", 1, 1, 3, 10), ("b", 3, 1, 10, 10)],  # x not contiguous
        [("a", 1, 1, 3, 10), ("b", 5, 1, 10, 10)],
    ],
)
def test_word_segmentation_raises(s):
    with pytest.raises(AssertionError):
        Segmentation.word(s, " ")


def test_word_segmentation_one_element():
    x = Segmentation.word([("a", 1, 1, 10, 10)], " ", include_spaces=True)
    assert x == [("a", 1, 1, 10, 10)]


def test_word_segmentation_with_spaces():
    s = [
        ("a", 1, 1, 2, 10),
        ("b", 3, 1, 3, 10),
        (" ", 4, 1, 5, 10),
        ("c", 6, 1, 800, 10),
    ]
    x = Segmentation.word(s, " ", include_spaces=True)
    e = [("ab", 1, 1, 3, 10), (" ", 4, 1, 5, 10), ("c", 6, 1, 800, 10)]
    assert x == e


def test_word_segmentation_without_spaces():
    s = [
        ("a", 1, 1, 2, 10),
        ("b", 3, 1, 3, 10),
        (" ", 4, 1, 5, 10),
        ("c", 6, 1, 800, 10),
    ]
    x = Segmentation.word(s, " ", include_spaces=False)
    e = [("ab", 1, 1, 3, 10), ("c", 6, 1, 800, 10)]
    assert x == e


def test_word_segmentation_space_at_beginning():
    s = [(" ", 1, 1, 2, 1), ("b", 3, 1, 3, 1), ("c", 4, 1, 5, 1)]
    x = Segmentation.word(s, " ", include_spaces=True)
    e = [(" ", 1, 1, 2, 1), ("bc", 3, 1, 5, 1)]
    assert x == e
    x = Segmentation.word(s, " ", include_spaces=False)
    e = [("bc", 3, 1, 5, 1)]
    assert x == e


def test_word_segmentation_space_at_end():
    s = [("a", 1, 1, 1, 1), ("b", 2, 1, 2, 1), (" ", 3, 1, 5, 1)]
    x = Segmentation.word(s, " ", include_spaces=True)
    e = [("ab", 1, 1, 2, 1), (" ", 3, 1, 5, 1)]
    assert x == e
    x = Segmentation.word(s, " ", include_spaces=False)
    e = [("ab", 1, 1, 2, 1)]
    assert x == e


class __TestSegmentation(Segmentation):
    def __init__(self, img_id, segm, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.img_id = img_id
        self.segm = re.escape(segm)

    @staticmethod
    def char(*_, **__):
        return [
            ("a", 1, 1, 2, 10),
            ("b", 3, 1, 3, 10),
            ("<space>", 4, 1, 5, 10),
            ("c", 6, 1, 800, 10),
        ]

    def write(self, value):
        assert re.match(self.img_id + self.segm, value)


@pytest.mark.parametrize(
    ["kwargs", "img_id", "segm"],
    [
        (
            {"include_img_ids": False},
            "",
            str(__TestSegmentation.char()),
        ),
        (
            {"separator": " --- "},
            r"va-\d+ --- ",
            str(__TestSegmentation.char()),
        ),
        (
            {"segmentation": "word"},
            r"va-\d+ ",
            str([("ab", 1, 1, 3, 10), ("<space>", 4, 1, 5, 10), ("c", 6, 1, 800, 10)]),
        ),
        pytest.param(
            {},
            "tr",
            "",
            marks=pytest.mark.xfail(reason="Check write called", strict=True),
        ),
    ],
)
@pytest.mark.parametrize("num_processes", (1, 2))
def test_segmentation_callback(tmpdir, num_processes, kwargs, img_id, segm):
    data_module = DummyMNISTLines(batch_size=2, va_n=12)
    trainer = DummyTrainer(
        default_root_dir=tmpdir,
        limit_test_batches=3,
        callbacks=[__TestSegmentation(img_id, segm, data_module.syms, **kwargs)],
        accelerator="ddp_cpu" if num_processes > 1 else None,
        num_processes=num_processes,
    )
    trainer.test(DummyEvaluator(), datamodule=data_module)
