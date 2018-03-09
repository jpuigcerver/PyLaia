from __future__ import absolute_import

import io
import json
import logging

# Inherit loglevels from Python's logging
DEBUG = logging.DEBUG
INFO = logging.INFO
WARN = logging.WARN
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL

BASIC_FORMAT = '%(asctime)s %(levelname)s %(name)s : %(message)s'
DETAILED_FORMAT = '%(asctime)s %(levelname)s %(name)s [%(pathname)s:%(lineno)d] : %(message)s'

# Root logger for Laia
_logger = logging.getLogger('laia')


def get_logger(name=None):
    if name is None:
        return _logger
    else:
        return logging.getLogger(name)


def basic_config(fmt=BASIC_FORMAT, level=INFO, filename=None,
                 filemode='a', log_also_to_stderr_level=ERROR):
    fmt = logging.Formatter(fmt)

    if filename:
        handler = logging.FileHandler(filename, filemode)
        handler.setFormatter(fmt)
        _logger.addHandler(handler)

        handler = logging.StreamHandler()
        handler.setLevel(log_also_to_stderr_level)
        handler.setFormatter(fmt)
        _logger.addHandler(handler)
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(fmt)
        _logger.addHandler(handler)

    _logger.setLevel(level)


def config(fmt=BASIC_FORMAT, level=INFO, filename=None, filemode='a',
           log_also_to_stderr_level=ERROR, config_dict=None):
    if config_dict:
        try:
            with io.open(config_dict, 'r') as f:
                config_dict = json.load(f)
            logging.config.dictConfig(config_dict)
        except Exception:
            basic_config()
            _logger.exception(
                'Logging configuration could not be parsed, using default')
    else:
        basic_config(fmt=fmt, level=level, filename=filename, filemode=filemode,
                     log_also_to_stderr_level=log_also_to_stderr_level)


def config_from_args(args, fmt=BASIC_FORMAT):
    config(config_dict=args.logging_config,
           filemode='w' if args.logging_overwrite else 'a',
           filename=args.logging_file,
           fmt=fmt,
           level=args.logging_level,
           log_also_to_stderr_level=args.logging_also_to_stderr)
