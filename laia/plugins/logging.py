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


def _get_logger(name=None):
    # Use 'laia' as the root logger
    return logging.getLogger(name if name else 'laia')


def basic_config(name=None, fmt=BASIC_FORMAT, level=INFO, filename=None,
                 filemode='a', log_also_to_stderr_level=ERROR):
    logger = _get_logger(name)
    fmt = logging.Formatter(fmt)

    handler = logging.StreamHandler()
    handler.setFormatter(fmt)
    if filename: handler.setLevel(log_also_to_stderr_level)
    logger.addHandler(handler)

    if filename:
        handler = logging.FileHandler(filename, filemode)
        handler.setFormatter(fmt)
        logger.addHandler(handler)

    logger.setLevel(level)


def config(name=None, fmt=BASIC_FORMAT, level=INFO, filename=None,
           filemode='a', log_also_to_stderr_level=ERROR, config_dict=None):
    if config_dict:
        try:
            with io.open(config_dict, 'r') as f:
                config_dict = json.load(f)
            # TODO: Where does this config come from?
            logging.config.dictConfig(config_dict)
        except Exception:
            basic_config(name)
            _get_logger(name).exception(
                'Logging configuration could not be parsed, using default')
    else:
        basic_config(name=name, fmt=fmt, level=level,
                     filename=filename, filemode=filemode,
                     log_also_to_stderr_level=log_also_to_stderr_level)


def config_from_args(args, fmt=BASIC_FORMAT):
    config(name=args.name,
           config_dict=args.logging_config,
           filemode='w' if args.logging_overwrite else 'a',
           filename=args.logging_file,
           fmt=fmt,
           level=args.logging_level,
           log_also_to_stderr_level=args.logging_also_to_stderr)


def log(level, msg, name=None, *args, **kwargs):
    logger = _get_logger(name)
    if logger.isEnabledFor(level):
        logger._log(level, msg.format(*args, **kwargs), (), ())


def debug(msg, name=None, *args, **kwargs):
    log(DEBUG, msg, name=name, *args, **kwargs)


def error(msg, name=None, *args, **kwargs):
    log(ERROR, msg, name=name, *args, **kwargs)


def info(msg, name=None, *args, **kwargs):
    log(INFO, msg, name=name, *args, **kwargs)


def warn(msg, name=None, *args, **kwargs):
    log(WARN, msg, name=name, *args, **kwargs)

def critical(msg, name=None, *args, **kwargs):
    log(CRITICAL, msg, name=name, *args, **kwargs)


def set_level(level, name=None):
    _get_logger(name).setLevel(level)
