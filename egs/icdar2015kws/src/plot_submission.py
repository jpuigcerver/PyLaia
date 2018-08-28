#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import argparse
import math
import os.path
import random

from PIL import Image, ImageDraw


def get_text():
    try:
        return raw_input()
    except NameError:
        return input()


def open_image_any_extension(img_dir, basename):
    prefix = os.path.join(img_dir, basename)
    valid_files = [
        prefix + ext
        for ext in [".png", ".jpg", ".tif", ".pgm"]
        if os.path.isfile(prefix + ext)
    ]
    assert valid_files, "Image for {!r} not found in {!r}".format(basename, img_dir)
    return Image.open(valid_files[0]).convert("RGBA")


def equalize_logp(spots):
    def cdf(iterable):
        cdf_v = []
        cdf_c = []
        for v in sorted(iterable):
            if not cdf_v or v != cdf_v[-1]:
                cdf_v.append(v)
                if not cdf_c:
                    cdf_c.append(1)
                else:
                    cdf_c.append(cdf_c[-1] + 1)
            else:
                cdf_c[-1] = cdf_c[-1] + 1

        return {v: c for v, c in zip(cdf_v, cdf_c)}, cdf_c[0]

    logp_values = [logp for page in spots for _, _, _, _, logp in spots[page]]
    if any([x is None for x in logp_values]):
        return spots

    N = len(logp_values)
    logp_cdf, mincdf = cdf(logp_values)
    equalized_spots = {}
    for page in spots:
        equalized_spots[page] = [
            (x, y, w, h, (logp_cdf[logp] - mincdf) / (N - mincdf))
            for x, y, w, h, logp in spots[page]
        ]
    return equalized_spots


def load_submission(f, min_relevance=None):
    if isinstance(f, (str, unicode)):
        f = io.open(f, "r", encoding="utf-8")

    index = {}
    for line in f:
        line = line.split()
        page, query, x, y, w, h = line[:6]
        logp = float(line[6]) if len(line) == 7 else None
        x, y, w, h = int(x), int(y), int(w), int(h)

        if min_relevance is None or logp > math.log(min_relevance):
            if page in index:
                index[page].append((query, x, y, w, h, logp))
            else:
                index[page] = [(query, x, y, w, h, logp)]
    f.close()
    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--min_relevance",
        type=float,
        default=None,
        help="Plot spots with this minimum relevance",
    )
    parser.add_argument("--page_scale", type=float, default=0.3, help="")
    parser.add_argument(
        "page_img_dir", type=str, help="Directory containing the page images"
    )
    parser.add_argument(
        "submission", type=argparse.FileType("r")
    )
    args = parser.parse_args()
    index = load_submission(args.submission, args.min_relevance)

    for page in sorted(index.keys()):
        page_img = open_image_any_extension(args.page_img_dir, page)
        page_w, page_h = page_img.size
        page_w = int(math.ceil(args.page_scale * page_w))
        page_h = int(math.ceil(args.page_scale * page_h))
        page_img = page_img.resize((page_w, page_h))

        boxes = Image.new("RGBA", page_img.size, (255, 255, 255, 0))
        drw = ImageDraw.Draw(boxes)
        query_colors = {}
        for (query, x, y, w, h, prob) in index[page]:

            def r():
                return random.randint(0, 255)

            def draw_rect(draw, coords, fill, outline, width=1):
                x, y, w, h = coords
                draw.rectangle([(x, y), (x + w, y + h)], fill=fill)
                draw.line(
                    [(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)],
                    fill=outline,
                    width=width,
                )

            if prob is not None:
                alpha = int(min(math.exp(prob) * 128, 128))
            else:
                alpha = 128

            x = int(math.ceil(args.page_scale * x))
            y = int(math.ceil(args.page_scale * y))
            w = int(math.ceil(args.page_scale * w))
            h = int(math.ceil(args.page_scale * h))

            if query in query_colors:
                red, green, blue = query_colors[query]
            else:
                red, green, blue = r(), r(), r()
                query_colors[query] = (red, green, blue)

            draw_rect(
                drw,
                (x, y, w, h),
                fill=(red, green, blue, alpha),
                outline=(red, green, blue, 255),
                width=3,
            )
            if prob is not None:
                print(page, query, x, y, w, h, prob)
            else:
                print(page, query, x, y, w, h)

        page_img = Image.alpha_composite(page_img, boxes)
        page_img.show()
        command = get_text().strip()
        if command == "n":
            break
        print(command)
