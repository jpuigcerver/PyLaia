from unittest.mock import patch

import numpy as np
import pytest

from laia.utils.visualize_segmentation import args, visualize


def test_visualize_segmentation(tmpdir):
    segmentation = tmpdir / "segmentation.txt"
    segmentation.write_text(
        "abc@[('a', 1, 1, 5, 10), ('sp', 6, 1, 8, 10), ('c', 9, 1, 15, 10)]",
        encoding="utf-8",
    )
    cmd_args = [
        "ignored",
        "img_path",
        str(segmentation),
        "abc",
        "--space=sp",
        "--separator=@",
    ]
    with patch("sys.argv", new=cmd_args):
        ns = args()
    imread_patch = patch("matplotlib.pyplot.imread", return_value=np.ones((10, 15)))
    show_patch = patch("matplotlib.pyplot.show")
    from matplotlib.pyplot import Axes

    axvspan_patch = patch.object(Axes, "axvspan")
    annotate_patch = patch.object(Axes, "annotate")
    with imread_patch as p1, show_patch as p2, axvspan_patch as p3, annotate_patch as p4:
        visualize(ns)
        assert p1.called
        assert p2.called
        assert p3.call_count == 3
        assert p4.call_count == 2


def test_visualize_segmentation_raises(tmpdir):
    segmentation = tmpdir / "segmentation.txt"
    segmentation.write_text("abc [('a', 1, 1, 5, 10)]", encoding="utf-8")
    cmd_args = ["ignored", "img_path", str(segmentation), "cba"]
    with patch("sys.argv", new=cmd_args):
        ns = args()
    with pytest.raises(ValueError):
        visualize(ns)
