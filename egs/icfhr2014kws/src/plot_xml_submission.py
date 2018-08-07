#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import argparse
import math
import os.path
import random
import xml.sax

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


class KWSContentHandler(xml.sax.ContentHandler):
    def __init__(self, page_img_dir, query_img_dir, min_relevance=None, page_scale=1.0):
        xml.sax.ContentHandler.__init__(self)
        self.page_img_dir = page_img_dir
        self.query_img_dir = query_img_dir
        self.min_relevance = min_relevance
        self.page_scale = page_scale
        self.queryid = None
        self.spots = None

    def startElement(self, name, attrs):
        if name == "Rel":
            self.queryid = attrs["queryid"]
            self.spots = {}
        elif name == "word":
            page = attrs["document"]
            x, y, w, h = attrs["x"], attrs["y"], attrs["width"], attrs["height"]
            logp = attrs["logp"] if "logp" in attrs else None

            x, y, w, h = int(x), int(y), int(w), int(h)
            logp = float(logp)

            if (
                self.min_relevance is not None
                and self.min_relevance > 0
                and logp < math.log(self.min_relevance)
            ):
                return

            if page in self.spots:
                self.spots[page].append((x, y, w, h, logp))
            else:
                self.spots[page] = [(x, y, w, h, logp)]

    def endElement(self, name):
        if name == "Rel":
            query_img = open_image_any_extension(self.query_img_dir, self.queryid)
            query_img.show()

            # self.spots = equalize_logp(self.spots)
            print(self.queryid)
            for page in self.spots:
                page_img = open_image_any_extension(self.page_img_dir, page)
                page_w, page_h = page_img.size
                page_w = int(math.ceil(self.page_scale * page_w))
                page_h = int(math.ceil(self.page_scale * page_h))
                page_img = page_img.resize((page_w, page_h))

                boxes = Image.new("RGBA", page_img.size, (255, 255, 255, 0))
                drw = ImageDraw.Draw(boxes)
                for (x, y, w, h, prob) in self.spots[page]:

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

                    x = int(math.ceil(self.page_scale * x))
                    y = int(math.ceil(self.page_scale * y))
                    w = int(math.ceil(self.page_scale * w))
                    h = int(math.ceil(self.page_scale * h))

                    red, green, blue = r(), r(), r()
                    draw_rect(
                        drw,
                        (x, y, w, h),
                        fill=(red, green, blue, alpha),
                        outline=(red, green, blue, 255),
                        width=3,
                    )
                    if prob is not None:
                        print(page, x, y, w, h, prob)
                    else:
                        print(page, x, y, w, h)

                page_img = Image.alpha_composite(page_img, boxes)
                page_img.show()
                command = get_text().strip()
                if command == "n":
                    break
                print(command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--min_relevance",
        type=float,
        default=0.0,
        help="Plot spots with this minimum relevance",
    )
    parser.add_argument("--page_scale", type=float, default=0.3, help="")
    parser.add_argument(
        "page_img_dir", type=str, help="Directory containing the page images"
    )
    parser.add_argument(
        "query_img_dir", type=str, help="Directory containing the page images"
    )
    parser.add_argument(
        "xml", type=argparse.FileType("r"), help="File containing the XML sumbission"
    )
    args = parser.parse_args()

    handler = KWSContentHandler(
        args.page_img_dir,
        args.query_img_dir,
        min_relevance=args.min_relevance,
        page_scale=args.page_scale,
    )
    xml.sax.parse(args.xml, handler)
