class Meter(object):
    def reset(self):
        pass

    @property
    def value(self):
        raise NotImplementedError('This method should be overriden.')
