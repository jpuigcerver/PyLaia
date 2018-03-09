from __future__ import absolute_import

import io
import json
import logging

BASIC_FORMAT = '%(asctime)s %(levelname)s %(name)s : %(message)s'
DETAILED_FORMAT = '%(asctime)s %(levelname)s %(name)s [%(pathname)s:%(lineno)d] : %(message)s'


def config(fmt=None, level=logging.INFO, filename=None, fileoverwrite=False,
           config_dict=None):
    def basic_config():
        logging.basicConfig(format=BASIC_FORMAT if fmt is None else fmt,
                            level=level,
                            filename=filename,
                            filemode='w' if fileoverwrite else 'a')

    if config_dict:
        try:
            with io.open(config_dict, 'r') as f:
                config_dict = json.load(f)
            logging.config.dictConfig(config_dict)
        except Exception:
            basic_config()
            logging.exception('Logging configuration could not be parsed')
    else:
        basic_config()
