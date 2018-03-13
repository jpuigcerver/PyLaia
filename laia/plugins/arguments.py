import argparse

from laia.plugins.arguments_types import (str2bool, NumberInClosedRange,
                                          NumberInOpenRange, str2loglevel)

_parser = None
_default_args = {
    'batch_size': (
        ('--batch_size',),
        {
            'type': NumberInClosedRange(type=int, vmin=1),
            'default': 8,
            'help': 'Batch size (must be >= 1)'
        }),
    'learning_rate': (
        ('--learning_rate',),
        {
            'type': NumberInOpenRange(type=float, vmin=0),
            'default': 0.0005,
            'help': 'Learning rate (must be > 0)'
        }),
    'momentum': (
        ('--momentum',),
        {
            'type': NumberInClosedRange(type=float, vmin=0),
            'default': 0,
            'help': 'Momentum (must be >= 0)'
        }),
    'gpu': (
        ('--gpu',),
        {
            'type': int,
            'default': 1,
            'help': 'Use this GPU (starting from 1)'
        }),
    'seed': (
        ('--seed',),
        {
            'type': int,
            'default': 0x12345,
            'help': 'Seed for random number generators'
        }),
    'final_fixed_height': (
        ('--final_fixed_height',),
        {
            'type': NumberInClosedRange(type=int, vmin=1),
            'default': 20,
            'help': 'Final height for the pseudo-images after the convolutions '
                    '(must be >= 1)'
        }),
    'max_epochs': (
        ('--max_epochs',),
        {
            'type': NumberInClosedRange(type=int, vmin=1),
            'help': 'Maximum number of training epochs'
        }),
    'max_updates': (
        ('--max_updates',),
        {
            'type': NumberInClosedRange(type=int, vmin=1),
            'help': 'Maximum number of parameter updates during training'
        }),
    'min_epochs': (
        ('--min_epochs',),
        {
            'type': NumberInClosedRange(type=int, vmin=1),
            'help': 'Minimum number of training epochs'
        }),
    'valid_cer_std_window_size': (
        ('--valid_cer_std_window_size',),
        {
            'type': NumberInClosedRange(type=int, vmin=2),
            'help': 'Use this number of epochs to compute the standard '
                    'deviation of the validation CER (must be >= 2)'
        }),
    'valid_cer_std_threshold': (
        ('--valid_cer_std_threshold',),
        {
            'type': NumberInOpenRange(type=float, vmin=0),
            'help': 'Stop training if the standard deviation of the validation '
                    'CER is below this threshold (must be > 0)'
        }),
    'valid_map_std_window_size': (
        ('--valid_map_std_window_size',),
        {
            'type': NumberInClosedRange(type=int, vmin=2),
            'help': 'Use this number of epochs to compute the standard '
                    'deviation of the validation Mean Average Precision (mAP) '
                    '(must be >= 2)'
        }),
    'valid_map_std_threshold': (
        ('--valid_map_std_threshold',),
        {
            'type': NumberInOpenRange(type=float, vmin=0),
            'help': 'Stop training if the standard deviation of the validation '
                    'Mean Average Precision (mAP) is below this threshold '
                    '(must be > 0)'
        }),
    'show_progress_bar': (
        ('--show_progress_bar',),
        {
            'type': str2bool,
            'nargs': '?',
            'const': True,
            'default': False,
            'help': 'Whether or not to show a progress bar for each epoch'
        }),
    'use_distortions': (
        ('--use_distortions',),
        {
            'type': str2bool,
            'nargs': '?',
            'const': True,
            'default': True,
            'help': 'Whether or not to use dynamic distortions to augment the '
                    'training data'
        }),
    'train_loss_std_threshold': (
        ('--train_loss_std_threshold',),
        {
            'type': NumberInOpenRange(type=float, vmin=0),
            'help': 'Stop training if the standard deviation of the training '
                    'loss is below this threshold (must be > 0)'
        }),
    'train_loss_std_window_size': (
        ('--train_loss_std_window_size',),
        {
            'type': NumberInClosedRange(type=int, vmin=2),
            'help': 'Use this number of epochs to compute the standard '
                    'deviation of the training loss (must be >= 2)'
        }),
    'num_samples_per_epoch': (
        ('--num_samples_per_epoch',),
        {
            'type': NumberInClosedRange(type=int, vmin=1),
            'help': 'Use this number of training examples randomly sampled '
                    'from the dataset in each epoch'
        }),
    'num_iterations_per_update': (
        ('--num_iterations_per_update',),
        {
            'default': 1,
            'type': NumberInClosedRange(type=int, vmin=1),
            'metavar': 'N',
            'help': 'Update parameters every N iterations'
        }),
    'weight_l2_penalty': (
        ('--weight_l2_penalty',),
        {
            'default': 0.0,
            'type': NumberInClosedRange(type=float, vmin=0),
            'help': 'Apply this L2 weight penalty to the loss function'
        }),
    'logging_also_to_stderr': (
        ('--logging_also_to_stderr',),
        {
            'default': 'ERROR',
            'type': str2loglevel,
            'help': 'If you are logging to a file, use this level for logging '
                    'also to stderr (use any of: debug, info, warning, error, '
                    'critical)',
        }),
    'logging_level': (
        ('--logging_level',),
        {
            'default': 'INFO',
            'type': str2loglevel,
            'help': 'Use this level for logging (use any of: debug, info, '
                    'warning, error, critical)',
        }),
    'logging_config': (
        ('--logging_config',),
        {
            'type': str,
            'help': 'Use this JSON file to configure the logging'
        }),
    'logging_file': (
        ('--logging_file',),
        {
            'type': str,
            'help': 'Write the logs to this file'
        }),
    'logging_overwrite': (
        ('--logging_overwrite',),
        {
            'type': str2bool,
            'nargs': '?',
            'const': True,
            'default': False,
            'help': 'If true, overwrite the logfile instead of appending it'
        }),
}


def _get_parser():
    global _parser
    if not _parser:
        _parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        add_defaults('logging_also_to_stderr', 'logging_config', 'logging_file',
                     'logging_level', 'logging_overwrite')
    return _parser


def add_defaults(*args, **kwargs):
    for arg in args:
        args_, kwargs_ = _default_args[arg]
        add_argument(*args_, **kwargs_)
    for arg, default_value in kwargs.items():
        args_, kwargs_ = _default_args[arg]
        kwargs_['default'] = default_value
        add_argument(*args_, **kwargs_)
    return _get_parser()


def add_argument(*args, **kwargs):
    _get_parser().add_argument(*args, **kwargs)
    return _parser


def args():
    return _get_parser().parse_args()
