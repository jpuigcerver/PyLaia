class Meter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        return self

    @property
    def value(self):
        raise NotImplementedError('This method should be overriden.')
