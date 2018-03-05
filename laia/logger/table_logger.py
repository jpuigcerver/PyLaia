from collections import OrderedDict

class TableLogger(object):
    def __init__(self, filename, fields):
        assert isinstance(fields, OrderedDict)
        self._filename = filename
        self._fields = fields

    @property
    def fields(self):
        return self._fields

    def add(self, **kwargs):
        for k, v in kwargs.iteritems():
            if k not in self._fields:
                raise KeyError('"%s" is not a registered field.' % k)
