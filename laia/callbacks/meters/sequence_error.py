import itertools
from typing import Iterable, List, Tuple

import textdistance

from laia.callbacks.meters.meter import Meter


def char_to_word_seq(
    seq: Iterable, delimiters: Iterable, reserved_char="\x1f"
) -> List[List]:
    """Convert a sequence of characters into a sequence of words.

    Args:
        seq (iterable): A list of symbols representing the
            characters in a sentence.
        delimiters (iterable): A set of symbols representing the word
            delimiters. Any sequence of characters
        reserved_char (str): Reserved character to split long delimiters

    Returns:
        A list of lists containing the characters that form each word.
        Delimiters are not included.
    """
    delimiter_set = set(delimiters)
    # Ugly hack to make delimiters with length > 1 work
    for d in delimiter_set:
        if isinstance(d, str) and len(d) > 1:
            seq = seq.replace(d, reserved_char)
    delimiter_set.add(reserved_char)
    return [
        list(y)
        for x, y in itertools.groupby(seq, key=lambda z: z in delimiter_set)
        if not x
    ]


class SequenceError(Meter):
    def __init__(self):
        super().__init__()
        self.num_errors = 0
        self.ref_length = 0

    def reset(self):
        self.num_errors = 0
        self.ref_length = 0
        return self

    def add(self, refs, hyps):
        num_errors, ref_length = SequenceError.distance(refs, hyps)
        self.num_errors += num_errors
        self.ref_length += ref_length
        return self

    @property
    def value(self) -> float:
        return SequenceError.error(self.num_errors, self.ref_length)

    @staticmethod
    def distance(refs, hyps) -> Tuple[int, int]:
        assert hasattr(refs, "__iter__") or hasattr(refs, "__getitem__")
        assert hasattr(hyps, "__iter__") or hasattr(hyps, "__getitem__")
        assert hasattr(refs, "__len__") and hasattr(hyps, "__len__")
        assert len(refs) == len(hyps)
        num_errors = sum(
            textdistance.levenshtein.distance(r, h) for r, h in zip(refs, hyps)
        )
        ref_length = sum(len(r) for r in refs)
        return num_errors, ref_length

    @staticmethod
    def compute(refs, hyps) -> float:
        num_errors, ref_length = SequenceError.distance(refs, hyps)
        return SequenceError.error(num_errors, ref_length)

    @staticmethod
    def error(num_errors, ref_length) -> float:
        if ref_length > 0:
            return float(num_errors) / float(ref_length)
        return float("nan")
