from __future__ import absolute_import

import numpy as np
from scipy.spatial.distance import pdist

import laia.logging as log
from laia.meters.meter import Meter
from laia.utils.accumulate import accumulate

_logger = log.get_logger(__name__)


class PairwiseAveragePrecisionMeter(Meter):
    r"""Compute Average Precision over pairs of objects using some metric.

    When a set of examples are added, its feature vectors and labels are
    given and stored in the ``PairwiseAveragePrecisionMeter`` object.
    The actual computation of the Average Precision will be done once the
    ``value`` property of the object is accessed.

    Then, the specified ``metric`` is used to measure all pairs of objects
    and these pairs are sorted in increasing order to compute both the
    Global and Mean Average Precision.

    *Important*: Self-pairs are ignored to compute the Average Precision.

    Args:
      metric (function, str): metric used to sort the pairs of objects
          (in increasing order). You can use the same metrics accepted by
          SciPy's ``pdist``.
      ignore_singleton (boolean, default: True): if true, ignore objects
          whose label is not shared with any other object.
    """

    def __init__(self, metric='euclidean', ignore_singleton=True):
        self._metric = metric
        self._features = []
        self._labels = []
        self._label_count = {}
        self._ignore_singleton = ignore_singleton
        self._gap, self._map = None, None
        # This is only to detect whether the metric is valid or not.
        _ = pdist([[1, 1], [1, 1]], metric)

    @property
    def logger(self):
        return _logger

    def reset(self):
        self._features = []
        self._labels = []
        self._label_count = {}
        self._gap, self._map = None, None
        return self

    def add(self, features, labels):
        r"""Add objects to the set used to compute the AP."""
        # If anything was computed, reset
        self._gap, self._map = None, None

        features = np.asarray(features)
        assert isinstance(labels, (list, tuple))

        self._features.append(features)
        self._labels.extend(labels)
        for c in labels:
            cnt = self._label_count.get(c, 0)
            self._label_count[c] = cnt + 1
        return self

    def _compute(self):
        # Concatenate all feature tensors (batches).
        all_features = np.concatenate(self._features)
        if self._ignore_singleton:
            mask = [i for i, c in enumerate(self._labels)
                    if self._label_count[c] > 1]
            all_features = all_features[mask]
            all_labels = [c for i, c in enumerate(self._labels)
                          if self._label_count[c] > 1]
        else:
            all_labels = self._labels

        n = all_features.shape[0]  # number of objects
        self.logger.debug('Compute Average Precision over {} samples', n)
        distances = pdist(all_features, self._metric)
        # Sort pairs of examples in increasing order
        inds = [(i, j) for i in range(n) for j in range(i + 1, n)]
        inds = [inds[k] for k in np.argsort(distances)]

        events = []
        events_i = [[] for _ in range(n)]
        for (i, j) in inds:
            if all_labels[i] == all_labels[j]:
                events.append(1.0)
                events_i[i].append(1.0)
                events_i[j].append(1.0)
            else:
                events.append(0.0)
                events_i[i].append(0.0)
                events_i[j].append(0.0)

        if events:
            # Compute Global and Mean Average Precision
            g_ap = self._compute_ap_ranked_events(events)
            aps = [self._compute_ap_ranked_events(e)
                   for e in events_i if len(e) > 0]
            m_ap = sum(aps) / len(aps)
            return g_ap, m_ap
        else:
            return 0.0, 0.0

    @classmethod
    def _compute_ap_ranked_events(cls, events):
        r"""Compute the average precision of a ranked list of events.

        Each event should be 1.0 if it is relevant or 0.0 if it is not
        relevant.
        """
        # Compute accumulated hits at each position in the ranking
        hits = list(accumulate(events))
        # Compute precision at each position in the ranking
        prec = [h / k for k, h in enumerate(hits, 1)]

        num = sum(p * r for p, r in zip(prec, events))
        den = hits[-1]
        if den > 0:
            return num / den
        else:
            # Handle the case where there aren't relevant events.
            return 0.0

    @property
    def value(self):
        r"""Return the Global and Mean Average Precision."""
        if self._gap is None or self._map is None:
            self._gap, self._map = self._compute()
        return self._gap, self._map
