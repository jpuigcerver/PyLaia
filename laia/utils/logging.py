import logging

BASIC_FORMAT = '%(asctime)s %(levelname)s %(name)s : %(message)s'
DETAILED_FORMAT = '%(asctime)s %(levelname)s %(name)s [%(pathname)s:%(lineno)d] : %(message)s'


def init(fmt=None):
    logging.basicConfig(format=BASIC_FORMAT if fmt is None else fmt)
