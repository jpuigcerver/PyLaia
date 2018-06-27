#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import print_function

import argparse
import math
import random
import re
from collections import namedtuple

from PIL import Image, ImageDraw
from typing import Union

from laia.utils.symbols_table import SymbolsTable

PositionMatch = namedtuple("PositionMatch", ("word", "position", "beg", "end", "logp"))
SegmentMatch = namedtuple("SegmentMatch", ("word", "beg", "end", "logp"))


def parse_position_match(s, syms=None):
    # type: (Union[str, unicode]) -> PositionMatch
    s = s.split()
    assert len(s) == 5

    if syms:
        w = " ".join([syms[int(x)] for x in s[0].split("_")])
    else:
        w = s[0]

    pos = int(s[1])
    beg = int(s[2])
    end = int(s[3])
    logp = float(s[4])

    return PositionMatch(word=w, position=pos, beg=beg, end=end, logp=logp)


def parse_segment_match(s, syms=None):
    # type: (Union[str, unicode]) -> SegmentMatch
    s = s.split()
    assert len(s) == 4

    if syms:
        w = " ".join([syms[int(x)] for x in s[0].split("_")])
    else:
        w = s[0]

    beg = int(s[1])
    end = int(s[2])
    logp = float(s[3])

    return SegmentMatch(word=w, beg=beg, end=end, logp=logp)


def match_scale_shift(
    match,  # type: Union[PositionMatch, SegmentMatch]
    scale,  # type: float
    shift,  # type: float
):
    # type: (...) -> Union[PositionMatch, SegmentMatch]
    beg = match.beg * scale + shift
    end = match.end * scale + shift
    if isinstance(match, PositionMatch):
        return PositionMatch(match.word, match.position, beg, end, match.logp)
    elif isinstance(match, SegmentMatch):
        return SegmentMatch(match.word, beg, end, match.logp)
    else:
        raise NotImplemented


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        type=int,
        default=5,
        help="Plot a single image containing this number of images",
    )
    parser.add_argument(
        "--symbols_table",
        type=argparse.FileType("r"),
        help="File containing the symbols table to map integers",
    )
    parser.add_argument(
        "--img_extension", type=str, default=".png", help="Image extension"
    )
    parser.add_argument(
        "--match_scale", type=int, default=1, help="Scale the match before plotting"
    )
    parser.add_argument(
        "--match_shift",
        type=int,
        default=0,
        help="Shift the match before plotting. Shift applied after scale.",
    )
    parser.add_argument(
        "--min_relevance",
        type=float,
        default=0.5,
        help="Plot objects with this minimum relevance",
    )
    parser.add_argument(
        "index_type",
        choices=("position", "segment"),
        help="Type of the KWS index to process",
    )
    parser.add_argument(
        "imgs_dir", type=str, help="Directory containing the indexed images"
    )
    parser.add_argument(
        "index", type=argparse.FileType("r"), help="File containing the KWS index"
    )
    args = parser.parse_args()
    syms = SymbolsTable(args.symbols_table) if args.symbols_table else None

    batch_images = []

    for sample in args.index:
        m = re.match(r"^([^ ]+) +(.+)$", sample)
        sample_id = m.group(1)
        print(sample_id)
        matches = m.group(2).split(";")
        # Parse index entries
        if args.index_type == "position":
            matches = [parse_position_match(m, syms) for m in matches]
            matches = sorted(matches, key=lambda x: x.position)
        elif args.index_type == "segment":
            matches = [parse_segment_match(m, syms) for m in matches]
            matches = sorted(matches, key=lambda x: x.beg)
        # Filter index entries
        matches = [m for m in matches if m.logp >= math.log(args.min_relevance)]
        # Scale matches
        matches = [
            match_scale_shift(m, args.match_scale, args.match_shift) for m in matches
        ]

        img = Image.open(args.imgs_dir + "/" + sample_id + args.img_extension)
        img = img.convert("RGBA")
        img_h, img_w = img.size

        boxes = Image.new("RGBA", img.size, (255, 255, 255, 0))
        drw = ImageDraw.Draw(boxes)

        for m in matches:

            def r():
                return random.randint(0, 255)

            alpha = int(round(255 * max(math.exp(m.logp) * 0.5, 0.5)))
            drw.rectangle([(m.beg, 0), (m.end, img_h)], fill=(r(), r(), r(), alpha))
            print(m)

        if len(batch_images) < args.batch_size:
            batch_images.append(Image.alpha_composite(img, boxes))
        else:
            max_w = max([im.size[0] for im in batch_images])
            sum_h = sum([im.size[1] for im in batch_images])

            sep_h = (len(batch_images) - 1) * 5
            container_im = Image.new(
                mode="RGBA", size=(max_w, sum_h + sep_h), color=(255, 255, 255, 0)
            )
            off_y = 0
            for im in batch_images:
                container_im.paste(im, (0, off_y))
                off_y += im.size[1] + 5
            # Clean batch
            batch_images = []
            container_im.show()
            raw_input()
