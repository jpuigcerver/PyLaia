from __future__ import absolute_import
from __future__ import division

import laia.logging as log
from laia.hooks.meters import Meter

_logger = log.get_logger(__name__)


class AveragePrecisionMeter(Meter):

    def __init__(self, desc_sort=True, exceptions_threshold=5):
        super(AveragePrecisionMeter, self).__init__(exceptions_threshold)
        self._desc_sort = desc_sort
        self._done = False
        self._ap = None
        self._matches = []

    @property
    def logger(self):
        return _logger

    def reset(self):
        self._done = False
        self._ap = None
        self._matches = []
        return self

    def add(self, tp, fp, fn, score=None):
        r"""Add objects to the set used to compute the AP."""
        self._done = False
        if score is None:
            score = -len(self._matches)
        # If anything was computed, reset
        self._ap = None
        self._matches.append((score, tp, fp, fn))
        return self

    @classmethod
    def _compute_ap_ranked_matches(cls, matches_tp_fp_fn):
        NH = 0  # Total number of hypotheses
        NR = 0  # Total number of references
        TTP = 0  # Total number of true positives
        acc = 0.0
        for (tp, fp, fn) in matches_tp_fp_fn:
            NH += tp + fp
            NR += tp + fn
            TTP += tp
            if tp > 0:
                acc += TTP / NH
        return acc / NR if NR > 0 else None

    @property
    def value(self):
        r"""Return the Average Precision."""
        if not self._done:
            self._done = True
            self._matches.sort(reverse=self._desc_sort)
            self._ap = self._compute_ap_ranked_matches(
                map(lambda x: x[1:], self._matches)
            )
        return self._ap
