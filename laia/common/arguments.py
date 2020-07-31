import argparse

from laia.common.arguments_types import (
    str2bool,
    NumberInClosedRange,
    NumberInOpenRange,
    str2loglevel,
)

_parser = None
_default_args = {
    "batch_size": (
        ("--batch_size",),
        {
            "type": NumberInClosedRange(type=int, vmin=1),
            "default": 8,
            "help": "Batch size (must be >= 1)",
        },
    ),
    "learning_rate": (
        ("--learning_rate",),
        {
            "type": NumberInOpenRange(type=float, vmin=0),
            "default": 0.0005,
            "help": "Learning rate (must be > 0)",
        },
    ),
    "momentum": (
        ("--momentum",),
        {
            "type": NumberInClosedRange(type=float, vmin=0),
            "default": 0,
            "help": "Momentum (must be >= 0)",
        },
    ),
    "weight_l2_penalty": (
        ("--weight_l2_penalty",),
        {
            "default": 0.0,
            "type": NumberInClosedRange(type=float, vmin=0),
            "help": "Apply this L2 weight penalty to the loss function",
        },
    ),
    "nesterov": (
        ("--nesterov",),
        {
            "type": str2bool,
            "nargs": "?",
            "const": True,
            "default": False,
            "help": "Whether or not to use Nesterov momentum",
        },
    ),
    "gpu": (
        ("--gpu",),
        {"type": int, "default": 1, "help": "Use this GPU (starting from 1)"},
    ),
    "seed": (
        ("--seed",),
        {
            "type": lambda x: int(x, 0),
            "default": 0x12345,
            "help": "Seed for random number generators",
        },
    ),
    "max_epochs": (
        ("--max_epochs",),
        {
            "type": NumberInClosedRange(type=int, vmin=1),
            "help": "Maximum number of training epochs",
        },
    ),
    "max_updates": (
        ("--max_updates",),
        {
            "type": NumberInClosedRange(type=int, vmin=1),
            "help": "Maximum number of parameter updates during training",
        },
    ),
    "min_epochs": (
        ("--min_epochs",),
        {
            "type": NumberInClosedRange(type=int, vmin=1),
            "help": "Minimum number of training epochs",
        },
    ),
    "valid_cer_std_window_size": (
        ("--valid_cer_std_window_size",),
        {
            "type": NumberInClosedRange(type=int, vmin=2),
            "help": "Use this number of epochs to compute the standard "
            "deviation of the validation CER (must be >= 2)",
        },
    ),
    "valid_cer_std_threshold": (
        ("--valid_cer_std_threshold",),
        {
            "type": NumberInOpenRange(type=float, vmin=0),
            "help": "Stop training if the standard deviation of the validation "
            "CER is below this threshold (must be > 0)",
        },
    ),
    "valid_map_std_window_size": (
        ("--valid_map_std_window_size",),
        {
            "type": NumberInClosedRange(type=int, vmin=2),
            "help": "Use this number of epochs to compute the standard "
            "deviation of the validation Mean Average Precision (mAP) "
            "(must be >= 2)",
        },
    ),
    "valid_map_std_threshold": (
        ("--valid_map_std_threshold",),
        {
            "type": NumberInOpenRange(type=float, vmin=0),
            "help": "Stop training if the standard deviation of the validation "
            "Mean Average Precision (mAP) is below this threshold "
            "(must be > 0)",
        },
    ),
    "show_progress_bar": (
        ("--show_progress_bar",),
        {
            "type": str2bool,
            "nargs": "?",
            "const": True,
            "default": False,
            "help": "Whether or not to show a progress bar for each epoch",
        },
    ),
    "use_distortions": (
        ("--use_distortions",),
        {
            "type": str2bool,
            "nargs": "?",
            "const": True,
            "default": False,
            "help": "Whether or not to use dynamic distortions to augment the "
            "training data",
        },
    ),
    "train_loss_std_threshold": (
        ("--train_loss_std_threshold",),
        {
            "type": NumberInOpenRange(type=float, vmin=0),
            "help": "Stop training if the standard deviation of the training "
            "loss is below this threshold (must be > 0)",
        },
    ),
    "train_loss_std_window_size": (
        ("--train_loss_std_window_size",),
        {
            "type": NumberInClosedRange(type=int, vmin=2),
            "help": "Use this number of epochs to compute the standard "
            "deviation of the training loss (must be >= 2)",
        },
    ),
    "train_samples_per_epoch": (
        ("--train_samples_per_epoch",),
        {
            "type": NumberInClosedRange(type=int, vmin=1),
            "help": "Use this number of training examples randomly sampled "
            "from the dataset in each epoch",
        },
    ),
    "valid_samples_per_epoch": (
        ("--valid_samples_per_epoch",),
        {
            "type": NumberInClosedRange(type=int, vmin=1),
            "help": "Use this number of validation examples randomly sampled "
            "from the dataset in each epoch",
        },
    ),
    "iterations_per_update": (
        ("--iterations_per_update",),
        {
            "default": 1,
            "type": NumberInClosedRange(type=int, vmin=1),
            "metavar": "N",
            "help": "Update parameters every N iterations",
        },
    ),
    "train_path": (
        ("--train_path",),
        {"type": str, "default": "", "help": "Save any files in this location"},
    ),
    "logging_also_to_stderr": (
        ("--logging_also_to_stderr",),
        {
            "default": "ERROR",
            "type": str2loglevel,
            "help": "If you are logging to a file, use this level for logging "
            "also to stderr (use any of: debug, info, warning, error, "
            "critical)",
        },
    ),
    "logging_level": (
        ("--logging_level",),
        {
            "default": "INFO",
            "type": str2loglevel,
            "help": "Use this level for logging (use any of: debug, info, "
            "warning, error, critical)",
        },
    ),
    "logging_config": (
        ("--logging_config",),
        {"type": str, "help": "Use this JSON file to configure the logging"},
    ),
    "logging_file": (
        ("--logging_file",),
        {"type": str, "help": "Write the logs to this file"},
    ),
    "logging_overwrite": (
        ("--logging_overwrite",),
        {
            "type": str2bool,
            "nargs": "?",
            "const": True,
            "default": False,
            "help": "If true, overwrite the logfile instead of appending it",
        },
    ),
    "print_args": (
        ("--print_args",),
        {
            "type": str2bool,
            "nargs": "?",
            "const": True,
            "default": True,
            "help": "If true, log to INFO the arguments passed to the program",
        },
    ),
    "save_checkpoint_interval": (
        ("--save_checkpoint_interval",),
        {
            "type": NumberInClosedRange(type=int, vmin=1),
            "default": None,
            "metavar": "N",
            "help": "Make checkpoints of the training process every N epochs",
        },
    ),
    "num_rolling_checkpoints": (
        ("--num_rolling_checkpoints",),
        {
            "type": NumberInClosedRange(type=int, vmin=1),
            "default": 2,
            "metavar": "N",
            "help": "Keep this number of last checkpoints during training",
        },
    ),
}


def _get_parser():
    global _parser
    if not _parser:
        _parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            conflict_handler="resolve",
        )
        add_defaults(
            "logging_also_to_stderr",
            "logging_config",
            "logging_file",
            "logging_level",
            "logging_overwrite",
            "print_args",
        )
    return _parser


def add_defaults(*args, **kwargs):
    for arg in args:
        args_, kwargs_ = _default_args[arg]
        add_argument(*args_, **kwargs_)
    for arg, default_value in kwargs.items():
        args_, kwargs_ = _default_args[arg]
        kwargs_["default"] = default_value
        add_argument(*args_, **kwargs_)
    return _get_parser()


def add_argument(*args, **kwargs):
    _get_parser().add_argument(*args, **kwargs)
    return _parser


def args() -> argparse.Namespace:
    a = _get_parser().parse_args()
    import laia.common.logging as log

    log.config_from_args(a)
    if a.print_args:
        import pprint

        log.get_logger(__name__).info("\n{}", pprint.pformat(vars(a)))
    return a
