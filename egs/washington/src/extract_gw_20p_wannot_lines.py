#!/usr/bin/env python3

import argparse
import io
import os
import os.path
from PIL import Image

def read_pages(fname):
    with io.open(fname, "r", encoding="utf-8") as f:
        for line in f:
            yield line.strip().split(".tif")[0]
    raise StopIteration

def read_boxes(fname, w, h):
    with io.open(fname, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            x1, x2, y1, y2, ly1, ly2 = [float(x) for x in line.split()]
            x1, x2 = round(x1 * (w - 1)), round(x2 * (w - 1))
            y1, y2 = round(y1 * (h - 1)), round(y2 * (h - 1))
            ly1, ly2 = round(ly1 * (h - 1)), round(ly2 * (h - 1))
            yield int(x1), int(x2), int(y1), int(y2), int(ly1), int(ly2)
    raise StopIteration

def read_words(fname, encoding="utf-8"):
    with io.open(fname, "r", encoding=encoding) as f:
        for line in f:
            yield line.strip()
    raise StopIteration

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("gw_20p_wannot",
                        help="Directory containing the gw_20p_wannot data")
    parser.add_argument("output_dir",
                        help="Output directpory for the text line images")
    args = parser.parse_args()

    words_iter = read_words(os.path.join(args.gw_20p_wannot, "annotations.txt"), encoding="iso-8859-1")

    os.makedirs(args.output_dir, exist_ok=True)

    for page in read_pages(os.path.join(args.gw_20p_wannot, "file_order.txt")):
        im = Image.open(os.path.join(args.gw_20p_wannot, page) + ".tif")
        im_w, im_h = im.size

        lines = {}
        for (x1, x2, y1, y2, ly1, ly2) in read_boxes(os.path.join(args.gw_20p_wannot, "{}_boxes.txt".format(page)), im_w, im_h):
            word = next(words_iter)
            if (ly1, ly2) in lines:
                lines[(ly1, ly2)].append((x1, x2, word))
            else:
                lines[(ly1, ly2)] = [(x1, x2, word)]

        line_keys = list(lines.keys())
        line_keys.sort()
        for n, (ly1, ly2) in enumerate(line_keys, 1):
            txt = u" ".join([word for _, _, word in lines[(ly1, ly2)]])
            lx1 = lines[(ly1, ly2)][0][0]
            lx2 = lines[(ly1, ly2)][-1][1]
            line_id = "{}-{:02d}".format(page[:3], n)
            line_im = im.crop((lx1, ly1, lx2, ly2))
            line_im.save(os.path.join(args.output_dir, "{}.png".format(line_id)))
            print(line_id, txt)
