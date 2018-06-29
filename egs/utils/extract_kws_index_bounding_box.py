#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import math
import random
import re
from collections import namedtuple
import os.path

import numpy as np
from typing import AnyStr, Dict, Tuple, Union, Optional
from typing.io import IO
import cv2

from laia.utils import SymbolsTable

logging.basicConfig(level=logging.WARNING)

TupleCoord = Tuple[float, float]
Parallelogram = Tuple[TupleCoord, ...]
ResizeInfo = namedtuple("ResizeInfo", ("scale", "offset_left", "offset_right", "offset_top", "offset_bottom"))

PositionMatch = namedtuple("PositionMatch", ("word", "position", "beg", "end", "logp"))
SegmentMatch = namedtuple("SegmentMatch", ("word", "beg", "end", "logp"))
Match = Union[PositionMatch, SegmentMatch]

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
    match,  # type: Match
    scale,  # type: float
    shift,  # type: float
):
    # type: (...) -> Match
    beg = match.beg * scale + shift
    end = match.end * scale + shift
    if isinstance(match, PositionMatch):
        return PositionMatch(match.word, match.position, beg, end, match.logp)
    elif isinstance(match, SegmentMatch):
        return SegmentMatch(match.word, beg, end, match.logp)
    else:
        raise NotImplementedError


def parse_paralellograms_file(f):
    # type: (IO[AnyStr]) -> Dict[str, Parallelogram]
    pgrams = {}
    for n, line in enumerate(f):
        line = line.split()
        if len(line) != 5:
            raise ValueError("Wrong parallelogram at line {}: wrong number of fields (found={}, expected={})".format(n, len(line), 5))
        try:
            sample_id = line[0]
            pgram = line[1:]
            pgram = [coord.split(",") for coord in pgram]
            pgram = [(float(x), float(y)) for x, y in pgram]
        except Exception as ex:
            raise ValueError("Wrong parallelogram at line {}: {}".format(n, ex))

        if sample_id in pgrams:
            logging.warning(
                "Sample %s was repeated in the parallelogram file", repr(sample_id)
            )

        pgrams[sample_id] = tuple(pgram)
    return pgrams

def parse_resize_info_file(f):
    # type: (IO[AnyStr]) -> Dict[str, ResizeInfo]
    resize_infos = {}
    for n, line in enumerate(f):
        line = line.split()
        if len(line) != 6:
            raise ValueError("Wrong resize info at line {}: wrong number of fields (found={}, expected={})".format(n, len(line), 6))
        try:
            sample_id = line[0]
            resize_info = [float(x) for x in line[1:]]
            resize_info = ResizeInfo(
                scale=resize_info[0],
                offset_left=resize_info[1],
                offset_right=resize_info[2],
                offset_top=resize_info[3],
                offset_bottom=resize_info[4]
            )
        except Exception as ex:
            raise ValueError("Wrong resize info at line {}: {}".format(n, ex))

        if sample_id in resize_infos:
            logging.warning(
                "Sample %s was repeated in the resize info file", repr(sample_id)
            )

        resize_infos[sample_id] = resize_info
    return resize_infos


def adjust_to_content(match, img, min_width_ratio=0.2):
    # type: (Match, np.ndarray, float) -> Match
    sum_rows = img.sum(axis=1)
    w = img.shape[0]
    x0, x1 = min(match.beg, 0), max(match.end, img.shape[1] - 1)

    def dilate_x(x, s):
        while x > 0 and x < w and sum_rows[x] > 0:
            x += s
        return x

    def erode_x(x, s):
        while x > 0 and x < w and sum_rows[x] == 0:
            x += s
        return x

    dilate_x(x0, -1)
    dilate_x(x1, +1)
    erode_x(x0, +1)
    erode_x(x1, +1)

    if (x1 - x0) / (match.end - match.beg) < 0.2:
        return match
    elif isinstance(match, PositionMatch):
        return PositionMatch(word=match.word, position=match.position, beg=x0, end=x1, logp=match.logp)
    elif isinstance(match, SegmentMatch):
        return SegmentMatch(word=match.word, beg=x0, end=x1, logp=match.logp)
    else:
        raise NotImplementedError


def undo_resize(x0, y0, x1, y1, resize_info):
    # type: (float, float, float, float, ResizeInfo) -> (float, float, float, float)
    x0 = x0 * resize_info.scale + resize_info.offset_left
    x1 = x1 * resize_info.scale + resize_info.offset_right
    y0 = y0 * resize_info.scale + resize_info.offset_top
    y1 = y1 * resize_info.scale + resize_info.offset_bottom
    return (x0, y0, x1, y1)


def transform_parallelogram(
    match_box,  # type: np.ndarray
    im_size,  # type: Tuple[int, int]
    line_pgram,  # type: Optional[Tuple[Tuple[float, float], ...]]
):
    # type: (...) -> np.ndarray
    line_box = np.asarray([
        [0, 0],
        [im_size[0], 0],
        [im_size[0], im_size[1]],
        [0, im_size[1]]
    ], dtype=np.float32)

    t = cv2.getPerspectiveTransform(src=line_box, dst=line_pgram)
    return np.dot(t, match_box)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adjust_to_content",
        action="store_true",
        help="Adjust the left and right coordinates of the word bounding boxes "
        "based on the content of the images",
    )
    parser.add_argument(
        "--img_extension", type=str, default=".png", help="Image extension"
    )
    parser.add_argument(
        "--match_global_scale",
        type=int,
        default=1,
        help="Scale the match before plotting"
    )
    parser.add_argument(
        "--match_global_shift",
        type=int,
        default=0,
        help="Shift the match before plotting. Shift applied after scale.",
    )
    parser.add_argument(
        "--resize_info_file",
        type=argparse.FileType("r"),
        help="File containing extra information used to resize images before the neural network"
    )
    parser.add_argument(
        "--page_id_regex",
        type=str,
        default=r"^([^.]+)\..*$",
        help="Regular expression used to extract the Page ID from the Line ID.",
    )
    parser.add_argument(
        "--symbols_table",
        type=argparse.FileType("r"),
        help="File containing the symbols table to map integers",
    )
    parser.add_argument(
        "--pgrams_file",
        type=argparse.FileType("r"),
        help="File containing the parallelograms",
    )
    parser.add_argument(
        "index_type",
        choices=("position", "segment"),
        help="Type of the KWS index to process",
    )
    parser.add_argument(
        "index_file", type=argparse.FileType("r"), help="File containing the KWS index"
    )
    parser.add_argument(
        "img_dir", type=str, help="Directory containing the processed images"
    )
    args = parser.parse_args()
    # Load symbols table
    syms = SymbolsTable(args.symbols_table) if args.symbols_table else None
    # Load parallelograms
    pgrams = parse_paralellograms_file(args.pgrams_file) if args.pgrams_file else None
    # Load resize info file
    resize_info = parse_resize_info_file(args.resize_info_file) if args.resize_info_file else None

    for n, sample in enumerate(args.index_file):
        m = re.match(r"^([^ ]+) +(.+)$", sample)
        if not m:
            raise ValueError("Wrong index entry at line{}".format(n))

        sample_id = m.group(1)

        # Get sample's parallelogram w.r.t. the original image
        sample_pgram = None
        if pgrams:
            if sample_id not in pgrams:
                logging.error("No parallelograms found for sample {!r}".format(sample_id))
            else:
                sample_pgram = pgrams[sample_id]

        # Parse sample matches
        matches = m.group(2).split(";")
        if args.index_type == "position":
            matches = [parse_position_match(m, syms) for m in matches]
        elif args.index_type == "segment":
            matches = [parse_segment_match(m, syms) for m in matches]
        # Scale matches
        matches = [
            match_scale_shift(m, args.match_global_scale, args.match_global_shift)
            for m in matches
        ]

        # Load processed image (NOT RESCALED)
        sample_img = cv2.imread(os.path.join(args.img_dir, sample_id, args.img_extension), 0)
        sample_img = cv2.adaptiveThreshold(sample_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # Get sample's resize info
        if resize_info:
            if sample_id not in resize_info:
                logging.error("No resize info found for sample {!r}".format(sample_id))
            else:
                sample_resize_info = resize_info[sample_id]

        # Process matches in the sample image
        for m in matches:
            # Adjust match to content
            if args.adjust_to_content:
                m = adjust_to_content(m, sample_img, min_width_ratio=0.3)

            # Resize bounding box
            if sample_resize_info:
                x0, y0, x1, y1 = undo_resize(m.beg, 0, m.end, sample_img.shape[1], sample_resize_info)
            else:
                x0, x1 = m.beg, m.end
                y0, y1 = 0, sample_img.shape[1]
