import itertools
from typing import Iterable, List

import textdistance

from laia.callbacks.meters import Meter


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
        for x, y in itertools.groupby(seq, lambda z: z in delimiter_set)
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
        assert hasattr(refs, "__iter__") or hasattr(refs, "__getitem__")
        assert hasattr(hyps, "__iter__") or hasattr(hyps, "__getitem__")
        assert hasattr(refs, "__len__") and hasattr(hyps, "__len__")
        assert len(refs) == len(hyps)
        for ref, hyp in zip(refs, hyps):
            self.num_errors += textdistance.levenshtein.distance(ref, hyp)
            self.ref_length += len(ref)
        return self

    @property
    def value(self) -> float:
        if self.ref_length > 0:
            return float(self.num_errors) / float(self.ref_length)
