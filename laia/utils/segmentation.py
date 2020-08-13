from typing import List, Tuple


def word_segmentation(
    char_segmentation: List[Tuple[str, int, int]],
    space: str,
    include_spaces: bool = True,
) -> List[Tuple[str, int, int]]:
    out = []
    w, w_start, w_end = "", 0, 0
    for i, (c, start, end) in enumerate(char_segmentation):
        if c == space:
            if w:
                out.append((w, w_start, w_end))
            if include_spaces:
                out.append((c, start, end))
            w = ""
            w_start = end
        else:
            if i == 0:
                w_start = start
            w += c
            w_end = end
        if i == len(char_segmentation) - 1 and w:
            out.append((w, w_start, end))
    return out
