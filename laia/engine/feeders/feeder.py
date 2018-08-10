class Feeder:
    """This class is used to feed data to a model or loss.

    During training or evaluation, a :class:`laia.engine.Engine` object
    will use feeders like this to feed data from a DataLoader into
    the model.

    Args:
      parent_feeder (callable, optional): parent feeder that should feed this.
          (default: None)
    """

    def __init__(self, parent_feeder=None):
        assert parent_feeder is None or callable(parent_feeder)
        self._parent_feeder = parent_feeder

    def __call__(self, batch):
        if self._parent_feeder:
            batch = self._parent_feeder(batch)
        return self._feed(batch)

    def _feed(self, batch):
        raise NotImplementedError("Abstract class.")


class ItemFeeder(Feeder):
    """Feed an element from a batch dictionary, by its key.

    Args:
      key: the key to use.
      parent_feeder (callable, optional): parent feeder that should feed this.
          (default: None)
    """

    def __init__(self, key, parent_feeder=None):
        super().__init__(parent_feeder)
        self._key = key

    def _feed(self, batch):
        assert self._key in batch, "Could not find batch[{}] for batch {}".format(
            self._key, batch
        )
        return batch[self._key]
