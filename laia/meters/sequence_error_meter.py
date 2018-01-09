import editdistance

from .meter import Meter

class SequenceErrorMeter(Meter):
    def __init__(self):
        super(SequenceErrorMeter, self).__init__()
        self.reset()

    def reset(self):
        self._num_errors = 0
        self._ref_length = 0

    def add(self, refs, hyps):
        assert isinstance(refs, (list, tuple))
        assert isinstance(hyps, (list, tuple))
        assert len(refs) == len(hyps)
        for ref, hyp in zip(refs, hyps):
            self._num_errors += editdistance.eval(ref, hyp)
            self._ref_length += len(ref)

    @property
    def value(self):
        return float(self._num_errors) / float(self._ref_length)
