import itertools
from typing import Iterable, List


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
