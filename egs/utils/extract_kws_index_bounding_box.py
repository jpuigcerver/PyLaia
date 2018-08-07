#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os.path
import re
from collections import namedtuple

import cv2
import numpy as np
from laia.utils import SymbolsTable
from typing import AnyStr, Dict, Tuple, Union
from typing.io import IO

logging.basicConfig(level=logging.WARNING)

TupleCoord = Tuple[float, float]
Parallelogram = Tuple[TupleCoord, ...]
ResizeInfo = namedtuple(
    "ResizeInfo",
    ("scale", "offset_left", "offset_right", "offset_top", "offset_bottom"),
)

PositionMatch = namedtuple("PositionMatch", ("word", "position", "beg", "end", "logp"))
SegmentMatch = namedtuple("SegmentMatch", ("word", "beg", "end", "logp"))
Match = Union[PositionMatch, SegmentMatch]


def parse_position_match(s, syms=None):
    # type: (Union[str, unicode], SymbolsTable) -> PositionMatch
    s = s.split()
    assert len(s) == 5

    if syms:
        w = "".join([syms[int(x)] for x in s[0].split("_")])
    else:
        w = s[0]

    pos = int(s[1])
    beg = int(s[2])
    end = int(s[3])
    logp = float(s[4])

    return PositionMatch(word=w, position=pos, beg=beg, end=end, logp=logp)


def parse_segment_match(s, syms=None):
    # type: (Union[str, unicode], SymbolsTable) -> SegmentMatch
    s = s.split()
    assert len(s) == 4

    if syms:
        w = "".join([syms[int(x)] for x in s[0].split("_")])
    else:
        w = s[0]

    beg = int(s[1])
    end = int(s[2])
    logp = float(s[3])

    return SegmentMatch(word=w, beg=beg, end=end, logp=logp)


def match_scale_shift(match, scale, shift):
    # type: (Match, Union[int, float], Union[int, float]) -> Match
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
            raise ValueError(
                "Wrong parallelogram at line {}: wrong number of fields "
                "(found={}, expected={})".format(n, len(line), 5)
            )
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

        pgrams[sample_id] = np.asarray(pgram, dtype=np.float32)
    return pgrams


def parse_resize_info_file(f):
    # type: (IO[AnyStr]) -> Dict[str, ResizeInfo]
    resize_infos = {}
    for n, line in enumerate(f):
        line = line.split()
        if len(line) != 6:
            raise ValueError(
                "Wrong resize info at line {}: wrong number of fields "
                "(found={}, expected={})".format(n, len(line), 6)
            )
        try:
            sample_id = line[0]
            resize_info = [float(x) for x in line[1:]]
            resize_info = ResizeInfo(
                scale=resize_info[0],
                offset_left=resize_info[1],
                offset_right=resize_info[2],
                offset_top=resize_info[3],
                offset_bottom=resize_info[4],
            )
        except Exception as ex:
            raise ValueError("Wrong resize info at line {}: {}".format(n, ex))

        if sample_id in resize_infos:
            logging.warning(
                "Sample %s was repeated in the resize info file", repr(sample_id)
            )

        resize_infos[sample_id] = resize_info
    return resize_infos


def undo_resize(x0, y0, x1, y1, resize_info):
    # type: (float, float, float, float, ResizeInfo) -> (float, float, float, float)
    x0 = x0 * resize_info.scale + resize_info.offset_left
    x1 = x1 * resize_info.scale + resize_info.offset_right
    y0 = y0 * resize_info.scale + resize_info.offset_top
    y1 = y1 * resize_info.scale + resize_info.offset_bottom
    return x0, y0, x1, y1


def undo_resize2(x0, x1, resize_info):
    x0 = x0 * resize_info.scale + resize_info.offset_left
    x1 = x1 * resize_info.scale + resize_info.offset_right
    return x0, x1


def transform_parallelogram(x, mat):
    # type: (np.ndarray, np.ndarray) -> np.ndarray
    assert x.shape == (4, 2)
    assert mat.shape == (3, 3)
    y = np.matmul(mat, np.append(x, [[1], [1], [1], [1]], axis=1).transpose())
    y = y.transpose()
    return y[:, :2] / y[:, 2].reshape((4, 1))


def corners_to_coord4(x0, y0, x1, y1, add_z=False):
    # type: (float, float, float, float) -> np.ndarray
    if add_z:
        return np.asarray(
            [[x0, y0, 1], [x1, y0, 1], [x1, y1, 1], [x0, y1, 1]], dtype=np.float32
        )
    else:
        return np.asarray([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)


def print_position_match(page_id, match, pgram, output_box=False):
    # type : (AnyStr, PositionMatch, np.ndarray, bool) -> None
    if output_box:
        x0 = int(round(pgram[:, 0].min()))
        x1 = int(round(pgram[:, 0].max()))
        y0 = int(round(pgram[:, 1].min()))
        y1 = int(round(pgram[:, 1].max()))
        print(
            "{} {} {} {} {} {} {} {}".format(
                page_id,
                x0,
                y0,
                x1 - x0,
                y1 - y0,
                match.word,
                match.position,
                match.logp,
            )
        )
    else:
        c = tuple(pgram.flatten())
        print(
            "{} {},{} {},{} {},{} {},{} {} {} {}".format(
                page_id,
                c[0],
                c[1],
                c[2],
                c[3],
                c[4],
                c[5],
                c[6],
                c[7],
                match.word,
                match.position,
                match.logp,
            )
        )


def print_segment_match(page_id, match, pgram, output_box=False):
    # type : (AnyStr, SegmentMatch, np.ndarray, bool) -> None
    if output_box:
        x0 = int(round(pgram[:, 0].min()))
        x1 = int(round(pgram[:, 0].max()))
        y0 = int(round(pgram[:, 1].min()))
        y1 = int(round(pgram[:, 1].max()))
        print(
            "{} {} {} {} {} {} {}".format(
                page_id, x0, y0, x1 - x0, y1 - y0, match.word, match.logp
            )
        )
    else:
        c = tuple(pgram.flatten())
        print(
            "{} {},{} {},{} {},{} {},{} {} {}".format(
                page_id,
                c[0],
                c[1],
                c[2],
                c[3],
                c[4],
                c[5],
                c[6],
                c[7],
                match.word,
                match.logp,
            )
        )


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
        "--global_scale", type=int, default=1, help="Scale the match before plotting"
    )
    parser.add_argument(
        "--global_shift",
        type=int,
        default=0,
        help="Shift the match before plotting. Shift applied after scale.",
    )
    parser.add_argument(
        "--resize_info_file",
        type=argparse.FileType("r"),
        help="File containing extra information used to resize images",
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
        "--output_bounding_box",
        action="store_true",
        help="Output the bounding box surrounding the match parallelogram",
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
        "pgrams_file",
        type=argparse.FileType("r"),
        help="File containing the parallelograms",
    )
    parser.add_argument(
        "img_dir", type=str, help="Directory containing the processed images"
    )
    args = parser.parse_args()
    # Load symbols table
    syms = SymbolsTable(args.symbols_table) if args.symbols_table else None
    # Load resize info file
    resize_info = (
        parse_resize_info_file(args.resize_info_file) if args.resize_info_file else None
    )
    # Load parallelograms
    pgrams = parse_paralellograms_file(args.pgrams_file)

    for n, sample in enumerate(args.index_file):
        m = re.match(r"^([^ ]+) +(.+)$", sample)
        if not m:
            raise ValueError("Wrong index entry at line{}".format(n))

        sample_id = m.group(1)
        pm = re.match(args.page_id_regex, sample_id)
        page_id = pm.group(1)

        # Parse sample matches
        matches = m.group(2).split(";")
        if args.index_type == "position":
            matches = [parse_position_match(m, syms) for m in matches]
        elif args.index_type == "segment":
            matches = [parse_segment_match(m, syms) for m in matches]
        # Scale matches
        matches = [
            match_scale_shift(m, args.global_scale, args.global_shift) for m in matches
        ]

        # Load image
        sample_img = cv2.imread(
            os.path.join(args.img_dir, sample_id + args.img_extension), 0
        )
        assert sample_img is not None
        _, sample_img = cv2.threshold(sample_img, 127, 255, cv2.THRESH_BINARY)
        im_x0, im_y0, im_x1, im_y1 = 0, 0, sample_img.shape[1], sample_img.shape[0]

        # Get sample's resize info
        sample_resize_info = None
        if resize_info:
            if sample_id not in resize_info:
                logging.error("No resize info found for sample {!r}".format(sample_id))
            else:
                sample_resize_info = resize_info[sample_id]
                im_x0, im_y0, im_x1, im_y1 = undo_resize(
                    im_x0, im_y0, im_x1, im_y1, sample_resize_info
                )

        # Get sample's parallelogram w.r.t. the original image
        if sample_id not in pgrams:
            logging.error("No parallelograms found for sample {!r}".format(sample_id))
            exit(1)

        sample_pgram = pgrams[sample_id]
        sample_box = corners_to_coord4(im_x0, im_y0, im_x1, im_y1)
        pgram_transform = cv2.getPerspectiveTransform(src=sample_box, dst=sample_pgram)

        # Process matches in the sample image
        for m in matches:
            # Adjust match to content
            if args.adjust_to_content:
                # TODO: Not implemented
                # x0, y0, x1, y1 = adjust_to_content(m, sample_img)
                x0, y0, x1, y1 = m.beg, 0, m.end, sample_img.shape[0]
            else:
                x0, y0, x1, y1 = m.beg, 0, m.end, sample_img.shape[0]

            # Resize bounding box
            if sample_resize_info:
                x0, y0, x1, y1 = undo_resize(x0, y0, x1, y1, sample_resize_info)

            match_pgram = corners_to_coord4(x0, y0, x1, y1)
            match_pgram = transform_parallelogram(match_pgram, pgram_transform)

            if isinstance(m, PositionMatch):
                print_position_match(page_id, m, match_pgram, args.output_bounding_box)
            elif isinstance(m, SegmentMatch):
                print_segment_match(page_id, m, match_pgram, args.output_bounding_box)
            else:
                raise NotImplementedError
