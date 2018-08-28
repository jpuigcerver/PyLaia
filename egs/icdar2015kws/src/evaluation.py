#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
__author__ = "Joan Puigcerver"
__email__ = "joapuipe@prhlt.upv.es"
__copyright__ = "Copyright 2015, Pattern Recognition and Human Language Technology"
__licence__ = "Public Domain"

import argparse
import bisect
import logging
import sys

from math import fabs, sqrt

LOG_FMT = '%(asctime)-15s %(levelname)s %(filename)s:%(lineno)d %(message)s'
logging.basicConfig(format = LOG_FMT)


class BoundingBox(object):
    """
    This class represents a bounding box object, defined by its top-left
    corner and its width and height.

    The most useful method is the relative_overlap() which computes the
    relative overlapping area between two bounding boxes.
    """

    def __init__(self, x, y, w, h):
        # Some assertions to be safe
        assert isinstance(x, int) and x >= 0
        assert isinstance(y, int) and y >= 0
        assert isinstance(w, int) and w >= 0
        assert isinstance(h, int) and h >= 0
        # Coordinates of the left-top corner, width and height
        self.x0, self.y0, self.w, self.h = x, y, w, h
        # These are the coordinates of the right-bottom corner
        self.x1, self.y1 = x + w - 1, y + h - 1
        # Area of the bounding box
        self.area = w * h
        # Bounding box center
        self.cx, self.cy = (self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0

    def __repr__(self):
        return repr(self.dict())

    def dict(self):
        return {'x': self.x0, 'y': self.y0, 'w': self.w, 'h': self.h}

    def equals(self, other):
        """Return true if two bounding boxes occupy the same location."""
        assert isinstance(other, BoundingBox)
        return self.x0 == other.x0 and self.y0 == other.y0 and \
               self.w == other.w and self.h == other.h

    def intersect_area(self, other):
        """Compute the area of the intersection between two bounding boxes."""
        assert isinstance(other, BoundingBox)
        h_overlap = max(0, min(self.x1 + 1, other.x1 + 1) - \
                        max(self.x0, other.x0))
        v_overlap = max(0, min(self.y1 + 1, other.y1 + 1) - \
                        max(self.y0, other.y0))
        return h_overlap * v_overlap

    def union_area(self, other):
        """Compute the area of the union between two bounding boxes."""
        assert isinstance(other, BoundingBox)
        return self.area + other.area - self.intersect_area(other)

    def relative_overlap(self, other):
        """Compute the relative overlapping area between two bounding boxes."""
        assert isinstance(other, BoundingBox)
        #return self.intersect_area(other) / float(self.area)
        return self.intersect_area(other) / float(self.union_area(other))

    def center_distance(self, other):
        """Compute the euclidean distance among the centers of two bounding
        boxes."""
        assert isinstance(other, BoundingBox)
        return sqrt((self.cx - other.cx)**2 + (self.cy - other.cy)**2)


class DetectedObject(BoundingBox):
    """
    This extends the BoundingBox class with a label attribute that
    identifies the detected object type.
    """
    def __init__(self, x, y, w, h, label, doc=None):
        super(DetectedObject, self).__init__(x, y, w, h)
        self.lbl = label
        self.doc = doc

    def __repr__(self):
        return repr(self.dict())

    def dict(self):
        d = super(DetectedObject, self).dict()
        d['lbl'] = self.lbl
        d['doc'] = self.doc
        return d

    def equals(self, other):
        """Return true if two detected objects are the same
        (same location, label and item)."""
        return super(DetectedObject, self).equals(other) and \
               self.lbl == other.lbl and self.doc == other.doc

class ObjectsReader(object):
    """
    This class is used to read a file containing a list of objects. Both
    reference and hypothesis objects are read using this class.

    Each line of the file consist of 6 fields, separated by spaces or tabs:
      1. Document ID
      2. Query ID / Label
      3. Bounding box X coordinate
      4. Bounding box Y coordinate
      5. Bounding box width
      6. Bounding box height

    Lines beginning with # are interpreted as comments.
    """
    def __init__(self, f = None):
        self.objs = []
        if f is not None:
            self.load(f)

    def load(self, f):
        try:
            self.objs = []
            if isinstance(f, str):
                f = open(f, 'r')
            ln = 0
            for line in f:
                ln += 1
                line = line.split()
                if line[0][0] == '#': continue
                if len(line) < 6:
                    logging.error('Missing fields in %s:%d', f.name, ln)
                    exit(1)
                doc, qry = line[0], line[1]
                x, y = int(line[2]), int(line[3])
                w, h = int(line[4]), int(line[5])
                self.objs.append(DetectedObject(x, y, w, h, qry, doc))
        except ValueError as e:
            logging.error('Wrong integer value in %s:%d', f.name, ln)
            exit(1)
        except IOError as e:
            logging.error('Error reading file %s: %s', f.name, e)
            exit(1)

    def __iter__(self):
        return self.objs.__iter__()

class Assessment(object):
    """
    This is an abstract class that computes quality statistics from a set of
    reference objects and detected objects.

    The general idea is that a spotting algorithm will generate a list of
    'detected_objects' (see DetectedObject class) and those will be matched
    against the reference list of objects ('reference_objects') to compute
    the basic statistics: true positives, false positives and false negatives.

    Depending on how the matching between the reference and the detected
    objects is done, the reported quality results can differ. That is the
    main reason to have an abstract Assessment class.

    See ThresholdedAssessment and ContinuousAssessment classes.
    """
    def __init__(self, reference_objects, detected_objects):
        # Sets of reference objects and detected objects (hypotheses)
        self.refs = reference_objects
        self.hyps = detected_objects
        # Different levels stats. These will be computed by subclasses
        self.global_stats = []
        self.lbl_stats = {}
        self.doc_stats = {}
        for obj in reference_objects:
            if not obj.doc in self.doc_stats:
                self.doc_stats[obj.doc] = []
            if not obj.lbl in self.lbl_stats:
                self.lbl_stats[obj.lbl] = []
        for obj in detected_objects:
            if not obj.doc in self.doc_stats:
                self.doc_stats[obj.doc] = []
            if not obj.lbl in self.lbl_stats:
                self.lbl_stats[obj.lbl] = []

    @staticmethod
    def compute_overlap_references(hyp, ref_objects):
        overlaps = []
        for ref in ref_objects:
            if ref.doc != hyp.doc or ref.lbl != hyp.lbl:
                continue
            overlap = ref.relative_overlap(hyp)
            distance = ref.center_distance(hyp)
            overlaps.append((overlap, -distance, ref))
        overlaps.sort(reverse=True)
        # Change the sign of the distances
        return map(lambda x: (x[0], -x[1], x[2]), overlaps)


class ThresholdedAssessment(Assessment):
    """
    An overlapping between a reference object and a detected object
    is relevant if the overlapping area surpasses a given threshold.

    More detailed explanation of the matching process:

    For all the reference objects and the detected objects, the overlapping
    area and the distance between the bounding box centers is computed.

    Each of these pairs (reference, detected) is sorted, first, according to the
    overlapping area (decreasing order) and, second, by the distance between
    centers (increasing order).

    A pair (reference, detected) is considered a hit, when both reference and
    detected objects have not been matched previously and the overlapping area
    is greater or equal to the given overlap_threshold.
    """
    def __init__(
            self, reference_objects, detected_objects, overlap_threshold=0.5):
        super(ThresholdedAssessment, self).__init__(
            reference_objects, detected_objects)
        self.__compute(overlap_threshold)

    def __compute(self, overlap_threshold):
        unmatched_refs = set(self.refs)
        for hyp in self.hyps:
            overlap_references = self.compute_overlap_references(
                hyp, unmatched_refs)
            matched_hyp = False
            for (overlap, _, ref) in overlap_references:
                if overlap >= overlap_threshold:
                    # True positive
                    self.global_stats.append((1, 0, 0))
                    self.doc_stats[ref.doc].append((1, 0, 0))
                    self.lbl_stats[ref.lbl].append((1, 0, 0))
                    unmatched_refs.remove(ref)
                    matched_hyp = True
                    break

            if not matched_hyp:
                # False positive
                self.global_stats.append((0, 1, 0))
                self.doc_stats[hyp.doc].append((0, 1, 0))
                self.lbl_stats[hyp.lbl].append((0, 1, 0))

        for ref in unmatched_refs:
            # False negative
            self.global_stats.append((0, 0, 1))
            self.doc_stats[ref.doc].append((0, 0, 1))
            self.lbl_stats[ref.lbl].append((0, 0, 1))


def ComputePrecisionAndRecall(events, interpolate = False):
    """
    Given a sorted list of events, compute teh Precision and Recall of each
    of the events.

    Events are tuples (tp, fp, fn) of float numbers in the range [0, 1], which
    indicate whether the given event is a hit (or true positive), a false
    positive, or was not detected at all (false negative).

    If the `interpolate' option is set to true, the Interpolated Precision
    is computed, instead of the regular Precision definition.

    The function returns the total number of relevant events, the total
    number of detected events and the precision and recall vectors.
    """
    # Number of events
    N = len(events)
    # Total number of relevant events
    TR = sum([tp + fn for (tp, _, fn) in events])
    # Precision and Recall at each point, sorted in increasing Recall
    Pr, Rc = [], []
    # Accumulated number of true positives and false positives
    TP, FP = 0.0, 0.0
    for (tp, fp, fn) in events:
        TP, FP = TP + tp, FP + fp
        Pr.append(TP / (TP + FP) if TP + FP > 0.0 else 0.0)
        Rc.append(TP / TR if TR > 0.0 else 0.0)
    # Interpolate precision
    if interpolate:
        for i in range(N - 1, 0, -1):
            Pr[i - 1] = max(Pr[i], Pr[i - 1])
    return TR, TP + FP, Pr, Rc


def ComputeMetrics(TR, Pr, Rc, events, trapezoid = False):
    """
    Compute different performance metrics from the precision and recall curve
    and the events list.

    `TR' is the total number of relevant events in the `events' list, `events'
    is a list of tuples (tp, fp, fn) as in the ComputePrecisionAndRecall
    function, and `Pr' and `Rc' are the Precision and Recall points defining
    the RP-curve. `trapezoid' can be used to use the `trapezoid' interpolation
    to approximate the integral used to compute the average precision.
    """
    N = len(events)
    AP, RPpos, RPdiff, F1Max, F1Min, PrMax, PrMin, RcMax, RcMin = \
        0.0, 0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0
    for i in range(N):
        AP += 0.5 * (Pr[i] + Pr[i-1]) * events[i][0] \
              if trapezoid and i > 0 else Pr[i] * events[i][0]
        rpd = fabs(Pr[i] - Rc[i])
        if rpd < RPdiff:
            RPdiff = rpd
            RPpos = i
        F1 = 2 * Pr[i] * Rc[i] / (Pr[i] + Rc[i]) if Pr[i] + Rc[i] > 0.0 else 0.0
        F1Max = max(F1Max, F1)
        F1Min = min(F1Min, F1)
        PrMax = max(PrMax, Pr[i])
        PrMin = min(PrMin, Pr[i])
        RcMax = max(RcMax, Rc[i])
        RcMin = min(RcMin, Rc[i])

    AP = AP / TR if TR > 0.0 else 0.0
    RP_p = Pr[RPpos] if len(Pr) > 0 else 0.0
    RP_r = Rc[RPpos] if len(Rc) > 0 else 0.0
    return AP, RP_p, RP_r, F1Max, F1Min, PrMax, PrMin, RcMax, RcMin


def InterpolateRecallPrecisionCurve(Rc, Pr, recall_points):
    """
    Given a Recall-Precision curve from a list of points for the recall and
    precision coordinates, interpolate the precision values of a list of recall
    points.

    This is useful when a Recall-Precision curve is obtained for each query and
    then all curves are averaged (the area under this curve is the mAP metric).
    """
    if len(Pr) != len(Rc):
        logging.error('Precision and recall curve points do not match!')
        logging.shutdown()
        exit(1)
    N = len(Pr)

    Pr2 = []
    for r in recall_points:
        lft = bisect.bisect_left(Rc, r)
        lft = lft - 1 if lft > 0 else 0
        rgt = bisect.bisect_right(Rc, r)
        rgt = rgt if rgt < N else N - 1
        r_lft, r_rgt = Rc[lft], Rc[rgt]
        if r_lft - r_rgt < 1E-5:
            Pr2.append(Pr[lft] * 0.5 + Pr[rgt] * 0.5)
        else:
            a = (r_rgt - r) / (r_rgt - r_lft)
            Pr2.append(Pr[lft] * a + Pr[rgt] * (1 - a))
    return Pr2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'ICDAR2015 Competition on KWS - Evaluation Toolkit')
    parser.add_argument(
        '--area-threshold', '-a', type=float, default=0.7,
        help='minimum overlapping area for hits')
    parser.add_argument(
        '--interpolate', '-i', action='store_true',
        help='interpolate precision')
    parser.add_argument(
        '--trapezoidal', '-t', action='store_true',
        help='use trapezoidal rule to compute average precision')
    parser.add_argument(
        '--global-rp', '-g', type=argparse.FileType('w'), default=None,
        help='output file with the global recall-precision curve points')
    parser.add_argument(
        '--mean-rp', '-m', type=argparse.FileType('w'), default=None,
        help='output file with the mean recall-precision curve points')
    parser.add_argument(
        '--mean-rp-points', '-p', type=int, default=100,
        help='number of points used in the mean recall-precision curve')
    parser.add_argument(
        'reference_file', type=argparse.FileType('r'),
        help='file containing the list of reference objects')
    parser.add_argument(
        'solution_file', type=argparse.FileType('r'),
        help='file containing the ranked list of detected objects')
    args = parser.parse_args()

    ref_file = ObjectsReader(args.reference_file)
    hyp_file = ObjectsReader(args.solution_file)

    assessment = ThresholdedAssessment(ref_file, hyp_file, args.area_threshold)

    # Compute Precision and Recall values for each event
    TR, TD, Pr, Rc = ComputePrecisionAndRecall(
        assessment.global_stats, args.interpolate)
    print 'Total relevant events =', int(TR)
    print 'Total detected events =', int(TD)
    if TR == 0:
        logging.error('No relevant events found in the ground truth')
        print 'AP = ?'
        print 'mAP = ?'
        print 'F1 max = ?'
        print 'F1 min = ?'
        print 'Precision max = ?'
        print 'Precision min = ?'
        print 'Recall max = ?'
        print 'Recall min = ?'
        print 'Precision at min. |R-P| = ?'
        print 'Recall at min. |R-P| = ?'
    else:
        # Global statistics
        AP, RP_p, RP_r, F1Max, F1Min, PrMax, PrMin, RcMax, RcMin = \
            ComputeMetrics(TR, Pr, Rc, assessment.global_stats, args.trapezoidal)

        # Compute mean AP, and mean Recall-Precision curve
        mRc = [i / (args.mean_rp_points - 1.0) for i in range(args.mean_rp_points)]
        mPr = [0.0] * args.mean_rp_points
        mAP, NQ = 0.0, 0
        for (lbl, events) in assessment.lbl_stats.iteritems():
            # Compute Precision and Recall for this particular query
            tr, td, pr, rc = ComputePrecisionAndRecall(
                events, args.interpolate)
            if tr == 0:
                logging.warning('Query "%s" was ignored in mAP computation', lbl)
            else:
                # Compute statistics
                ap = ComputeMetrics(tr, pr, rc, events, args.trapezoidal)[0]
                mAP += ap
                NQ +=1
                # Compute interpolated recall-precision curve
                if args.mean_rp:
                    pr = InterpolateRecallPrecisionCurve(rc, pr, mRc)
                    mPr = map(lambda i: pr[i] + mPr[i], range(args.mean_rp_points))
        # Average mAP sum
        mAP = mAP / NQ if NQ > 0 else 0.0

        # Save Global RP-curve
        if args.global_rp:
            for i in range(len(Pr)):
                args.global_rp.write('%.9e %.9e\n' % (Rc[i], Pr[i]))
            args.global_rp.close()

        # Save Mean RP-curve
        if args.mean_rp and NQ > 0:
            mPr = map(lambda x: x / NQ, mPr)
            for i in range(len(mPr)):
                args.mean_rp.write('%.9e %.9e\n' % (mRc[i], mPr[i]))
            args.mean_rp.close()

        # Print metrics
        print 'AP =', AP
        print 'mAP =', mAP
        print 'F1 max =', F1Max
        print 'F1 min =', F1Min
        print 'Precision max =', PrMax
        print 'Precision min =', PrMin
        print 'Recall max =', RcMax
        print 'Recall min =', RcMin
        print 'Precision at min. |R-P| =', RP_p
        print 'Recall at min. |R-P| =', RP_r
