class Meter(object):
    """A meter tracks something on its :attr:`~.Meter.value`.

    Meters are used for measuring times, computing running averages, and
    many other metrics.

    See for instance:
      - :class:`~laia.meters.RunningAverageMeter`
      - :class:`~laia.meters.SequenceErrorMeter`
      - :class:`~laia.meters.TimeMeter`
    """

    @property
    def value(self):
        """Access the latest value of the meter."""
        raise NotImplementedError('This method should be overriden.')
