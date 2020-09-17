import logging
import os
import sys

from pytorch_lightning.utilities import rank_zero_only
from tqdm.auto import tqdm

# Inherit loglevels from Python's logging
DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL

BASIC_FORMAT = "%(asctime)s %(levelname)s %(name)s : %(message)s"
DETAILED_FORMAT = (
    "%(asctime)s %(levelname)s %(name)s [%(pathname)s:%(lineno)d] : %(message)s"
)


class TqdmStreamHandler(logging.StreamHandler):
    """
    This handler uses tqdm.write to log so
    logging messages don't break the tqdm bar.
    """

    def __init__(self, level=logging.NOTSET):
        super().__init__()
        self.setLevel(level)

    def emit(self, record):
        if record.levelno < self.level:
            return
        try:
            msg = self.format(record)
            tqdm.write(msg, file=sys.stderr)
            self.flush()
        except Exception:  # noqa
            self.handleError(record)


class FormatMessage:
    def __init__(self, fmt, *args, **kwargs):
        self.fmt = fmt
        self.args = args
        self.kwargs = kwargs

    def __str__(self):
        return str(self.fmt).format(*self.args, **self.kwargs)


class Logger(logging.Logger):
    def __init__(self, name, level=logging.NOTSET):
        super().__init__(name, level)

    def _log(self, level, msg, args, **kwargs):
        if "exc_info" in kwargs:
            exc_info = kwargs["exc_info"]
            del kwargs["exc_info"]
        else:
            exc_info = None

        if "extra" in kwargs:
            extra = kwargs["extra"]
            del kwargs["extra"]
        else:
            extra = None

        if args or kwargs:
            msg = FormatMessage(msg, *args, **kwargs)

        super()._log(level=level, msg=msg, args=(), exc_info=exc_info, extra=extra)


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
    logging._acquireLock()  # noqa
    backup_class = logging.getLoggerClass()
    logging.setLoggerClass(Logger)
    # Use 'laia' as the root logger
    logger = logging.getLogger(name if name else "laia")
    logging.setLoggerClass(backup_class)
    logging._releaseLock()  # noqa
    return logger


# Laia root logger
root = get_logger()
# pytorch-lightning logger
lightning = logging.getLogger("lightning")
lightning.handlers = []
# warnings logger
warnings = logging.getLogger("py.warnings")


def handle_exception(exc_type, exc_value, exc_traceback):
    # https://stackoverflow.com/a/16993115
    if not len(root.handlers) or all(
        isinstance(h, logging.NullHandler) for h in root.handlers
    ):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    root.critical("Uncaught exception:", exc_info=(exc_type, exc_value, exc_traceback))


def set_exception_handler(func=sys.__excepthook__):
    sys.excepthook = func


def capture_warnings():
    import warnings

    def format_warning(msg, category, filename, lineno, *_):
        return f"{filename}:{lineno}: {category.__name__}: {msg}"

    warnings.formatwarning = format_warning
    logging.captureWarnings(True)


def config(
    fmt=BASIC_FORMAT,
    level=INFO,
    filename=None,
    filemode="a",
    logging_also_to_stderr=ERROR,
    exception_handling_fn=handle_exception,
):
    capture_warnings()

    set_exception_handler(func=exception_handling_fn)
    fmt = logging.Formatter(fmt)

    # log to stderr on master only
    if rank_zero_only.rank == 0:
        handler = TqdmStreamHandler(level=level)
        handler.setFormatter(fmt)
        if filename:
            handler.setLevel(logging_also_to_stderr)
        root.addHandler(handler)
        lightning.addHandler(handler)
        warnings.addHandler(handler)

    if filename:
        if rank_zero_only.rank > 0:
            filename += f".rank{rank_zero_only.rank}"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        handler = logging.FileHandler(filename, filemode)
        handler.setFormatter(fmt)
        root.addHandler(handler)
        lightning.addHandler(handler)
        warnings.addHandler(handler)

    root.setLevel(level)


def config_from_args(args, fmt=BASIC_FORMAT):
    config(
        filemode="w" if args.logging_overwrite else "a",
        filename=args.logging_file,
        fmt=fmt,
        level=args.logging_level,
        logging_also_to_stderr=args.logging_also_to_stderr,
    )


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
