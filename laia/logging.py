from __future__ import absolute_import

import io
import json
import logging
import sys

try:
    import tqdm
except ImportError:
    tqdm = None

# Inherit loglevels from Python's logging
DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL

BASIC_FORMAT = '%(asctime)s %(levelname)s %(name)s : %(message)s'
DETAILED_FORMAT = '%(asctime)s %(levelname)s %(name)s [%(pathname)s:%(lineno)d] : %(message)s'


class TqdmStreamHandler(logging.StreamHandler):
    """This handler uses tqdm.write to log so logging
        messages don't break the tqdm bar."""

    def __init__(self, level=logging.NOTSET):
        super(TqdmStreamHandler, self).__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg, file=sys.stderr)
            self.flush()
        except Exception:
            self.handleError(record)


class FormatMessage(object):
    def __init__(self, fmt, *args, **kwargs):
        self.fmt = fmt
        self.args = args
        self.kwargs = kwargs

    def __str__(self):
        return str(self.fmt).format(*self.args, **self.kwargs)


class Logger(logging.Logger):
    def __init__(self, name, level=logging.NOTSET):
        super(Logger, self).__init__(name, level)

    def _log(self, level, msg, args, **kwargs):
        if 'exc_info' in kwargs:
            exc_info = kwargs['exc_info']
            del kwargs['exc_info']
        else:
            exc_info = None

        if 'extra' in kwargs:
            extra = kwargs['extra']
            del kwargs['extra']
        else:
            extra = None

        if args or kwargs:
            msg = FormatMessage(msg, *args, **kwargs)

        super(Logger, self)._log(level=level, msg=msg, args=(),
                                 exc_info=exc_info, extra=extra)


def get_logger(name=None):
    """Create/Get a Laia logger.

    The logger is an object of the class :class:`~.Logger` use the new string
    formatting, and accepts keyword arguments.

    Arguments:
        name (str) : name of the logger to get. If `None`, the root logger
            for Laia will be returned (`laia`).

    Returns:
        A :obj:`~.Logger` object.
    """
    logging._acquireLock()
    backup_class = logging.getLoggerClass()
    logging.setLoggerClass(Logger)
    # Use 'laia' as the root logger
    logger = logging.getLogger(name if name else 'laia')
    logging.setLoggerClass(backup_class)
    logging._releaseLock()
    return logger


# Laia root logger
root = get_logger()


def basic_config(fmt=BASIC_FORMAT, level=INFO, filename=None,
                 filemode='a', logging_also_to_stderr=ERROR):
    fmt = logging.Formatter(fmt)

    handler = TqdmStreamHandler() if tqdm else logging.StreamHandler()
    handler.setFormatter(fmt)
    if filename: handler.setLevel(logging_also_to_stderr)
    root.addHandler(handler)

    if filename:
        handler = logging.FileHandler(filename, filemode)
        handler.setFormatter(fmt)
        root.addHandler(handler)

    root.setLevel(level)


def config(fmt=BASIC_FORMAT, level=INFO, filename=None,
           filemode='a', logging_also_to_stderr=ERROR, config_dict=None):
    if config_dict:
        try:
            with io.open(config_dict, 'r') as f:
                config_dict = json.load(f)
            logging.config.dictConfig(config_dict)
        except Exception:
            basic_config()
            root.exception(
                'Logging configuration could not be parsed, using default')
    else:
        basic_config(fmt=fmt, level=level,
                     filename=filename, filemode=filemode,
                     logging_also_to_stderr=logging_also_to_stderr)


def config_from_args(args, fmt=BASIC_FORMAT):
    config(config_dict=args.logging_config,
           filemode='w' if args.logging_overwrite else 'a',
           filename=args.logging_file,
           fmt=fmt,
           level=args.logging_level,
           logging_also_to_stderr=args.logging_also_to_stderr)


def log(level, msg, *args, **kwargs):
    root.log(level, msg, *args, **kwargs)


def debug(msg, *args, **kwargs):
    root.debug(msg, *args, **kwargs)


def error(msg, *args, **kwargs):
    root.error(msg, *args, **kwargs)


def info(msg, *args, **kwargs):
    root.info(msg, *args, **kwargs)


def warning(msg, *args, **kwargs):
    root.warning(msg, *args, **kwargs)


def critical(msg, *args, **kwargs):
    root.critical(msg, *args, **kwargs)


def set_level(level):
    root.setLevel(level)


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        root.info('Laia stopped')
        return
    root.exception('Uncaught Laia exception:', exc_info=(
        exc_type, exc_value, exc_traceback))


sys.excepthook = handle_exception
