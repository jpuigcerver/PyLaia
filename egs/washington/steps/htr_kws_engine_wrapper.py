from __future__ import absolute_import

from collections import Counter

from laia.engine.htr_engine_wrapper import HtrEngineWrapper
from laia.hooks import action
from laia.decoders.ctc_nbest_path_decoder import CTCNBestPathDecoder
from laia.hooks.meters import Meter
from laia.utils.accumulate import accumulate

import pywrapfst as fst


class NBestPairwiseAveragePrecisionMeter(Meter):
    def __init__(self, ignore_singleton=True, exceptions_threshold=5):
        super(NBestPairwiseAveragePrecisionMeter, self).__init__(
            exceptions_threshold)
        self._ignore_singleton = ignore_singleton
        self._nbests = []
        self._labels = []
        self._gap, self._map = None, None

    def reset(self):
        self._nbests = []
        self._labels = []
        self._gap, self._map = None, None
        return self

    def add(self, nbests, labels):
        r"""Add objects to the set used to compute the AP."""
        # If anything was computed, reset
        self._gap, self._map = None, None
        self._nbests.extend(nbests)
        self._labels.extend(labels)
        return self

    def _compute(self):
        if self._ignore_singleton:
            labels_counts = Counter(self._labels)
            labels_nbest = [(label, nbest)
                            for label, nbest in zip(self._labels, self._nbests)
                            if labels_counts[label] > 1]
        else:
            labels_nbest = list(zip(self._labels, self._nbests))

        events = []
        for i, (la, nba) in enumerate(labels_nbest):
            for j in range(i + 1, len(labels_nbest)):
                lb, nbb = labels_nbest[j]
                # Compute match score
                match_score = 0.0
                for (lkh_a, path_a) in nba:
                    for (lkh_b, path_b) in nbb:
                        if path_a == path_b:
                            match_score += lkh_a + lkh_b
                events.append((match_score, i, j, 1.0 if la == lb else 0.0))

        events.sort(reverse=True)
        events_i = [[] for _ in range(len(labels_nbest))]
        for k, ev in enumerate(events):
            _, i, j, hit = ev
            events[i] = hit
            events_i[i].append(hit)
            events_i[j].append(hit)

        for i, ei in enumerate(events_i):
            ei.sort(reverse=True)
            ei = [rel for _, rel in ei]
            events_i[i] = ei

        if events:
            # Compute Global and Mean Average Precision
            g_ap = self._compute_ap_ranked_events(events)
            aps = [self._compute_ap_ranked_events(e)
                   for e in events_i if len(e) > 0]
            m_ap = sum(aps) / len(aps)
            return g_ap, m_ap
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
        # Handle the case where there aren't relevant events.
        return 0.0

    @property
    def value(self):
        r"""Return the Global and Mean Average Precision."""
        if self._gap is None or self._map is None:
            self._gap, self._map = self._compute()
        return self._gap, self._map


class HtrKwsEngineWrapper(HtrEngineWrapper):
    def __init__(self, train_engine, valid_engine, nbest=10):
        super(HtrKwsEngineWrapper, self).__init__(train_engine, valid_engine)
        self._nbest_decoder = CTCNBestPathDecoder(nbest)
        self._valid_ap_meter = NBestPairwiseAveragePrecisionMeter()


    @action
    def _valid_reset_meters(self):
        super(HtrKwsEngineWrapper, self)._valid_reset_meters()

    @action
    def _valid_update_meters(self, batch_output, batch_target):
        super(HtrKwsEngineWrapper, self)._valid_update_meters(batch_output,
                                                              batch_target)
        batch_nbests = self._nbest_decoder(batch_output)
        self._valid_ap_meter.add(batch_nbests,
                                 ['_'.join(map(str, t)) for t in batch_target])

    def _prepare_epoch_summary(self):
        fmt, params = super(HtrKwsEngineWrapper, self)._prepare_epoch_summary()
        fmt += ', VA gAP = {valid_ap.value[0]:5.1%}'
        fmt += ', VA mAP = {valid_ap.value[1]:5.1%}'
        params['valid_ap'] = self._valid_ap_meter
        return fmt, params




