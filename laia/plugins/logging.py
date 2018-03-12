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


class Message(object):
    def __init__(self, fmt, *args, **kwargs):
        self.fmt = fmt
        self.args = args
        self.kwargs = kwargs

    def __str__(self):
        return self.fmt.format(*self.args, **self.kwargs)


class StyleAdapter(logging.LoggerAdapter):
    def __init__(self, logger, extra=None):
        super(StyleAdapter, self).__init__(logger, extra or {})
        self.logger = logger

    def log(self, level, msg, *args, **kwargs):
        if self.isEnabledFor(level):
            msg, kwargs = self.process(msg, kwargs)
            self.logger._log(level, Message(msg, *args, **kwargs), ())


def get_logger(name=None):
    # Use 'laia' as the root logger
    return StyleAdapter(logging.getLogger(name if name else 'laia'))


def basic_config(fmt=BASIC_FORMAT, level=INFO, filename=None,
                 filemode='a', log_also_to_stderr_level=ERROR):
    logger = get_logger().logger
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


def config(fmt=BASIC_FORMAT, level=INFO, filename=None,
           filemode='a', log_also_to_stderr_level=ERROR, config_dict=None):
    if config_dict:
        try:
            with io.open(config_dict, 'r') as f:
                config_dict = json.load(f)
            logging.config.dictConfig(config_dict)
        except Exception:
            basic_config()
            get_logger().exception(
                'Logging configuration could not be parsed, using default')
    else:
        basic_config(fmt=fmt, level=level,
                     filename=filename, filemode=filemode,
                     log_also_to_stderr_level=log_also_to_stderr_level)


def config_from_args(args, fmt=BASIC_FORMAT):
    config(config_dict=args.logging_config,
           filemode='w' if args.logging_overwrite else 'a',
           filename=args.logging_file,
           fmt=fmt,
           level=args.logging_level,
           log_also_to_stderr_level=args.logging_also_to_stderr)


def log(level, msg, *args, **kwargs):
    get_logger().log(level, msg, *args, **kwargs)


def debug(msg, *args, **kwargs):
    log(DEBUG, msg, *args, **kwargs)


def error(msg, *args, **kwargs):
    log(ERROR, msg, *args, **kwargs)


def info(msg, *args, **kwargs):
    log(INFO, msg, *args, **kwargs)


def warn(msg, *args, **kwargs):
    log(WARN, msg, *args, **kwargs)


def critical(msg, *args, **kwargs):
    log(CRITICAL, msg, *args, **kwargs)


def set_level(level):
    get_logger().setLevel(level)
