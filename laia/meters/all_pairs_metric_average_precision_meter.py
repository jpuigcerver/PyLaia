from __future__ import absolute_import

import numpy as np
import torch

from functools import reduce
from laia.meters.meter import Meter
from laia.utils import accumulate
from scipy.spatial.distance import pdist


class AllPairsMetricAveragePrecisionMeter(Meter):
    r"""Compute Average Precision over all pairs of features using some metric.

    When an object is called, the functions ``features_fn'' and ``class_fn''
    will be used to extract the matrix of features (as many rows as objects in
    the batch, and as many columns as dimensions), and the class of the object.

    These are stored in the object repeatedly, until the ``value'' property of
    the object is read. Then, the metric is used to measure all pairs of objects
    and these pairs are sorted in increasing order to compute the
    Average Precision.

    Both Global and Mean Average Precision are computed.

    *Important*: Self-pairs are ignored to compute the Average Precision.

    Args:
      features_fn (function): function that returns the features from the
          ``**kwargs`` when called.
      class_fn (function): function that returns the classes from the
          ``**kwargs`` when called.
      metric (function, str): metric used to sort the pairs of objects
          (in increasing order). You can use the same metrics accepted by
          SciPy's ``pdist``.
      ignore_singleton (boolean, default: True): if true, ignore objects
          whose class is not shared with any other object.
    """

    def __init__(self, features_fn, class_fn, metric='euclidean',
                 ignore_singleton=True):
        assert callable(features_fn)
        assert callable(class_fn)
        self._features_fn = features_fn
        self._class_fn = class_fn
        self._metric = metric
        self._features = []
        self._classes = []
        self._class_count = {}
        self._ignore_singleton = ignore_singleton
        self._gap, self._map = None, None
        # This is only to detect whether the metric is valid or not.
        _ = pdist([[1, 1], [1, 1]], metric)

    def reset(self):
        self._features = []
        self._classes = []
        self._class_count = {}
        self._gap, self._map = None, None

    def __call__(self, **kwargs):
        r"""Add objects to the set used to compute the AP.

        The ``**kwargs`` arguments will be passed to ``features_fn`` and
        ``class_fn`` to extract the vector of features and the class for each
        example in the batch.
        """
        # If anything was computed, reset
        self._gap, self._map = None, None

        features = self._features_fn(**kwargs)
        classes = self._class_fn(**kwargs)

        assert torch.is_tensor(features)
        assert features.dim() == 2
        assert isinstance(classes, (list, tuple))

        self._features.append(features.numpy())
        self._classes.extend(classes)
        for c in classes:
            cnt = self._class_count.get(c, 0)
            self._class_count[c] = cnt + 1

    def _compute(self):
        # Concatenate all feature tensors (batches).
        all_features = np.concatenate(self._features)
        if self._ignore_singleton:
            mask = [self._class_count[c] > 1 for c in self._classes]
            all_features = all_features[mask]

        n = all_features.shape[0]   # number of objects
        distances = pdist(all_features, self._metric)
        # Sort pairs of examples in increasing order
        inds = [(i, j) for i in range(n) for j in range(i + 1, n)]
        inds = [inds[k] for k in np.argsort(distances)]

        events = []
        events_i = [[] for i in range(n)]
        for (i, j) in inds:
            if self._classes[i] == self._classes[j]:
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
            aps = [self._compute_ap_ranked_events(e) for e in events_i if len(e) > 0]
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
