class Loss(object):

    def __init__(self):
        self._loss = None

    def __call__(self, output, target):
        raise NotImplementedError("This method must be overriden")

    @property
    def loss(self):
        return self._loss
