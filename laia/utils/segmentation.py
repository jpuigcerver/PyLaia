from typing import List, Tuple, Optional


def char_segmentation(
    txt: str, seg: List[int], height: int, width: Optional[int] = None
) -> List[Tuple[str, int, int, int, int]]:
    assert len(txt) + 2 == len(seg)
    if width:
        # Scale the width
        max_pos = seg[-1]
        seg = [(x * width // max_pos) for x in seg]
    # Convert (0-based start, non-inclusive end)
    # to (1-based start, inclusive end)
    seg[0] += 1
    seg[-1] += 1
    return [
        # (value, p1=(x, y), p2=(x, y))
        (txt[j], seg[j], 1, seg[j + 1] - 1, height)
        for j in range(len(txt))
    ]


def word_segmentation(
    segmentation: List[Tuple[str, int, int, int, int]],
    space: str,
    include_spaces: bool = True,
) -> List[Tuple[str, int, int, int, int]]:
    pairs = list(zip(segmentation, segmentation[1:]))
    assert all(s1[3] + 1 == s2[1] for s1, s2 in pairs)
    assert all(s1[2] == s2[2] for s1, s2 in pairs)
    assert all(s1[4] == s2[4] for s1, s2 in pairs)
    out = []
    w, w_x1, w_x2 = "", None, None
    for i, (c, x1, y1, x2, y2) in enumerate(segmentation):
        if c == space:
            if w:
                out.append((w, w_x1, y1, w_x2, y2))
            if include_spaces:
                out.append((c, x1, y1, x2, y2))
            w = ""
            w_x1 = x2 + 1
        else:
            if i == 0:

                w_x1 = x1
            w += c
            w_x2 = x2
        if i == len(segmentation) - 1 and w:
            out.append((w, w_x1, y1, x2, y2))
    return out
