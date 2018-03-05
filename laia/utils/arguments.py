import argparse

_parser = None
_default_args = {
    'batch_size': (('--batch_size',),
                   {'type': int, 'default': 8, 'help': 'Batch size'}),
    'learning_rate': (('--learning_rate',),
                      {'type': float, 'default': 0.0005, 'help': 'Learning rate'}),
    'momentum': (('--momentum',),
                 {'type': float, 'default': 0, 'help': 'Momentum'}),
    'gpu': (('--gpu',),
            {'type': int, 'default': 1, 'help': 'Use this GPU (starting from 1)'}),
    'seed': (('--seed',),
             {'type': int, 'default': 0x12345, 'help': 'Seed for random number generators'}),
    'final_fixed_height': (('--final_fixed_height',),
                           {'type': int, 'default': 20,
                            'help': 'Final height for the pseudo-images after the convolutions'}),
    'max_epochs': (('--max_epochs',),
                   {'type': int, 'help': 'Maximum number of training epochs'}),
    'cer_stddev_values': (('--cer_stddev_values',),
                          {'type': int,
                           'help': 'Compute the standard deviation of the CER over this number of epochs'}),
    'cer_stddev_threshold': (('--cer_stddev_threshold',),
                             {'type': float,
                              'help': 'Stop training if the standard deviation of the CER falls below this threshold'})
}


def _get_parser():
    global _parser
    if not _parser:
        _parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    return _parser


def add_defaults(*args):
    # If called without arguments add all
    if len(args) == 0:
        for args_, kwargs_ in _default_args.values():
            add_argument(*args_, **kwargs_)
    # Otherwise add only those given
    else:
        for arg in args:
            args_, kwargs_ = _default_args[arg]
            add_argument(*args_, **kwargs_)
    return _parser


def add_argument(*args, **kwargs):
    _get_parser().add_argument(*args, **kwargs)


def args():
    return _get_parser().parse_args()


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
