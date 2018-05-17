#!/usr/bin/env python

from __future__ import absolute_import

import argparse

import torch.nn as nn
from laia.models.htr.gated_crnn import GatedCRNN
from laia.plugins import ModelSaver
from laia.plugins.arguments import add_argument, args, add_defaults
from laia.plugins.arguments_types import NumberInClosedRange, TupleList, \
    str2bool
from laia.utils import SymbolsTable

if __name__ == '__main__':
    add_defaults('train_path')
    add_argument('num_input_channels', type=NumberInClosedRange(int, vmin=1),
                 help='Number of channels of the input images')
    add_argument('syms', type=argparse.FileType('r'),
                 help='Symbols table mapping from strings to integers')
    add_argument('--cnn_num_features', type=NumberInClosedRange(int, vmin=1),
                 nargs='+', default=[8, 16, 32, 64, 128],
                 help='Number of features in each convolutional layer')
    add_argument('--cnn_kernel_size', type=TupleList(int, dimensions=2),
                 nargs='+', default=[3, (2, 4), 3, (2, 4), 3],
                 help='Kernel size of each convolution. '
                      'Use a list of integers, or a list of strings '
                      'representing tuples (height, width). '
                      'e.g: "(2, 4)" "(2, 4)"')
    add_argument('--cnn_stride', type=TupleList(int, dimensions=2),
                 nargs='+', default=[1, (2, 4), 1, (2, 4), 1],
                 help='Stride of each convolution. '
                      'Use a list of integers, or a list of strings '
                      'representing tuples (height, width). '
                      'e.g: "(2, 4)" "(2, 4)"')
    add_argument('--cnn_add_gating', type=str2bool, nargs='+',
                 default=[False, True, True, True, False],
                 help='Whether or not to add a gating mechanism after each '
                      'convolution layer')
    add_argument('--cnn_poolsize', type=TupleList(int, dimensions=2), nargs='+',
                 default=None,
                 help='Maxpool size after each convolution layer. '
                      'Use a list of integers, or a list of strings '
                      'representing tuples (height, width). '
                      'e.g: "(2, 4)" "(2, 4)"')
    add_argument('--cnn_activations', default=['Tanh'] * 5, nargs='+',
                 choices=['ReLU', 'Tanh', 'LeakyReLU'],
                 help='Type of the activation function after each convolution')
    add_argument('--sequencer_type', default='maxpool-1', type=str,
                 help='Specify how to convert an image to a sequence')
    add_argument('--columnwise', type=str2bool, default=True,
                 help='If true, the image will be processed as a sequence '
                      'column-wise')
    add_argument('--rnn_hidden_size', default=128,
                 type=NumberInClosedRange(int, vmin=1),
                 help='Number of units the recurrent layers')
    add_argument('--rnn_layers', default=2,
                 type=NumberInClosedRange(int, vmin=1),
                 help='Number of recurrent layers')
    add_argument('--rnn_type', choices=['LSTM', 'GRU'], default='LSTM',
                 help='Type of the recurrent layers')
    add_argument('--rnn_dropout', default=0.5,
                 type=NumberInClosedRange(float, vmin=0, vmax=1),
                 help='Dropout before and after the recurrent layers')
    args = args()

    num_output_symbols = len(SymbolsTable(args.syms))

    ModelSaver(args.train_path, args.filename) \
        .save(GatedCRNN,
              in_channels=args.num_input_channels,
              num_outputs=num_output_symbols,
              cnn_num_features=args.cnn_num_features,
              cnn_kernel_sizes=args.cnn_kernel_size,
              cnn_strides=args.cnn_stride,
              cnn_add_gating=args.cnn_add_gating,
              cnn_poolsize=args.cnn_poolsize,
              cnn_activation=[getattr(nn, act) for act in args.cnn_activations],
              sequencer=args.sequencer,
              columnwise=args.columnwise,
              rnn_hidden_size=args.rnn_hidden_size,
              rnn_num_layers=args.rnn_layers,
              rnn_type=getattr(nn, args.rnn_type),
              rnn_bidirectional=True,
              rnn_dropout=args.dropout)
