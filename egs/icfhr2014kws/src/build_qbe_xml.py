#!/usr/bin/env python

import argparse
from itertools import izip, product

import math
from scipy.misc import logsumexp


def load_queries(f):
    def pairwise(iterable):
        a = iter(iterable)
        return izip(a, a)

    queries = []
    for n, line in enumerate(f):
        line = line.split()
        if (len(line) - 1) % 2 != 0:
            raise ValueError("Wrong nbest at line {}".format(n))

        queries.append((line[0], [(w, float(p)) for w, p in pairwise(line[1:])]))

    return queries


def load_kws_index(f, filter=None):
    index = {}
    for n, line in enumerate(f):
        line = line.split()
        if len(line) == 8:
            # positional index
            page = line[0]
            x, y, w, h = [int(z) for z in line[1:5]]
            word = line[5]
            logp = float(line[7])
        elif len(line) == 7:
            # segment index
            page = line[0]
            x, y, w, h = [int(z) for z in line[1:5]]
            word = line[5]
            logp = float(line[6])
        else:
            raise ValueError("Wrong index format at line {}".format(n))

        if filter and word not in filter:
            continue

        if word in index:
            pages = index[word]
            if page in pages:
                pages[page].append((x, y, w, h, logp))
            else:
                pages[page] = [(x, y, w, h, logp)]
        else:
            index[word] = {page: [(x, y, w, h, logp)]}

    return index


def load_biggest_regions(f):
    regions = {}
    for n, line in enumerate(f):
        line = line.split()
        if len(line) != 5:
            raise ValueError("Wrong biggest region at line {}".format(n))
        regions[line[0]] = tuple([int(x) for x in line[1:]])
    return regions


def iou(bb1, bb2):
    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[0] + bb1[2], bb2[0] + bb2[2])
    y_bottom = min(bb1[1] + bb1[3], bb2[1] + bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = bb1[2] * bb1[3]
    bb2_area = bb1[2] * bb1[3]

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    return intersection_area / float(bb1_area + bb2_area - intersection_area)


def combine_boxes(bboxes):
    assert bboxes
    log_x = math.log1p(bboxes[0][0]) + bboxes[0][4]
    log_y = math.log1p(bboxes[0][1]) + bboxes[0][4]
    log_w = math.log1p(bboxes[0][2]) + bboxes[0][4]
    log_h = math.log1p(bboxes[0][3]) + bboxes[0][4]
    log_s = bboxes[0][4]
    for x, y, w, h, logp in bboxes[1:]:
        log_x = logsumexp([log_x, math.log1p(x) + logp])
        log_y = logsumexp([log_y, math.log1p(y) + logp])
        log_w = logsumexp([log_w, math.log1p(w) + logp])
        log_h = logsumexp([log_h, math.log1p(h) + logp])
        log_s = logsumexp([log_s, logp])

    result = (
        math.expm1(log_x - log_s),
        math.expm1(log_y - log_s),
        math.expm1(log_w - log_s),
        math.expm1(log_h - log_s),
        log_s,
    )
    return result


class UnionFind(object):
    def __init__(self, bounding_boxes):
        self._parent = [-1] * len(bounding_boxes)
        self._bounding_boxes = bounding_boxes
        self._num_parents = len(bounding_boxes)

    @property
    def num_parents(self):
        return self._num_parents

    def find(self, a):
        while self._parent[a] >= 0:
            a = self._parent[a]
        return a

    def merge(self, a, b):
        pa = self.find(a)
        pb = self.find(b)
        if pa == pb:
            return pa
        if self._parent[pa] > self._parent[pb]:
            self._parent[pa] = pb
            self._parent[pb] -= 1
            self._num_parents -= 1
            return pa
        else:
            self._parent[pb] = pa
            self._parent[pa] -= 1
            self._num_parents -= 1
            return pb

    def compress(self):
        for i in range(len(self._parent)):
            pi = self.find(i)
            if pi != i:
                self._parent[i] = pi

    def iter_parents(self):
        for i, pi in enumerate(self._parent):
            if pi < 0:
                yield i
        raise StopIteration

    def iter_groups(self):
        self.compress()
        groups = {}
        for i, pi in enumerate(self._parent):
            if pi < 0:
                pi = i

            if pi in groups:
                groups[pi].append(i)
            else:
                groups[pi] = [i]
        for g in groups.values():
            yield sorted(g)
        raise StopIteration

    def __getitem__(self, item):
        return self._bounding_boxes[item]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--biggest_regions", type=argparse.FileType("r"))
    parser.add_argument("--scale_w", type=float, default=1)
    parser.add_argument("--scale_h", type=float, default=1)
    parser.add_argument("kws_index", type=argparse.FileType("r"))
    parser.add_argument("query_nbest", type=argparse.FileType("r"))
    args = parser.parse_args()

    queries = load_queries(args.query_nbest)
    index = load_kws_index(
        args.kws_index, filter={w for _, nbest in queries for w, p in nbest}
    )
    biggest_regions = (
        load_biggest_regions(args.biggest_regions) if args.biggest_regions else None
    )

    print('<?xml version="1.0" encoding="utf-8"?>')
    print(
        '<RelevanceListings xmlns:xsd="http://www.w3.org/2001/XMLSchema" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
    )
    for query, nbest in queries:
        spots = {}
        for (word, logp1) in nbest:
            if word not in index:
                continue

            for page in index[word]:
                if page in spots:
                    spots[page].extend(
                        [
                            (x, y, w, h, logp1 + logp2)
                            for x, y, w, h, logp2 in index[word][page]
                        ]
                    )
                else:
                    spots[page] = [
                        (x, y, w, h, logp1 + logp2)
                        for x, y, w, h, logp2 in index[word][page]
                    ]

        # Combine spots
        combined_spots = []
        for page in spots:
            ufs = UnionFind(spots[page])
            while True:
                parents = [p for p in ufs.iter_parents()]
                merge = False
                for p1, p2 in product(parents, parents):
                    if p1 < p2:
                        bb1 = ufs[p1]
                        bb2 = ufs[p2]
                        if iou(bb1, bb2) > 0.7:
                            ufs.merge(p1, p2)
                            merge = True
                if not merge:
                    break

            for group_spots in ufs.iter_groups():
                x, y, w, h, logp = combine_boxes([ufs[item] for item in group_spots])
                combined_spots.append((logp, page, x, y, w, h))

        print('<Rel queryid="{}">'.format(query))
        for logp, page, x, y, w, h in sorted(combined_spots, reverse=True):
            if biggest_regions:
                rx, ry, rw, rh = biggest_regions[page]
                if x < rx or x > rx + rw or y < ry or y > ry + rh:
                    continue
                offset_x = rx
                offset_y = ry
            else:
                offset_x = 0
                offset_y = 0

            dw = w * args.scale_w - w
            dh = h * args.scale_h - h
            print(
                '<word document="{}" x="{}" y="{}" width="{}" height="{}" '
                'logp="{}" />'.format(
                    page,
                    int(round(x - offset_x - dw / 2.0)),
                    int(round(y - offset_y - dh / 2.0)),
                    int(round(w + dw / 2.0)),
                    int(round(h + dh / 2.0)),
                    logp,
                )
            )
        print("</Rel>")
    print("</RelevanceListings>")
