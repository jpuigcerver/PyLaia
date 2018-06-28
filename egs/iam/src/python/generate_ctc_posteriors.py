#!/usr/bin/env python

from __future__ import print_function

import argparse

import torch
from dortmund_utils import build_ctc_model
from laia.data import ImageDataLoader
from laia.data import TextImageFromTextTableDataset
from laia.common.arguments import add_argument, add_defaults, args
from laia.utils import ImageToTensor, TextToTensor
from laia.utils.symbols_table import SymbolsTable


def dump_output_matrix(img_ids, outputs, fileout, add_boundary_blank):
    for img_id, output in zip(img_ids, outputs):
        output = output.cpu()
        print('{} ['.format(img_id), file=fileout)
        if add_boundary_blank:
            print('{:.10g}'.format(0.0), file=fileout, end=' ')
            for k in range(1, output.size(1)):
                print('{:.10g}'.format(-1e30), file=fileout, end=' ')
            print('', file=fileout)
        for t in range(output.size(0)):
            for k in range(output.size(1)):
                print('{:.10g}'.format(float(output[t, k])), file=fileout, end=' ')
            print('', file=fileout)
        if add_boundary_blank:
            print('{:.10g}'.format(0.0), file=fileout, end=' ')
            for k in range(1, output.size(1)):
                print('{:.10g}'.format(-1e30), file=fileout, end=' ')
            print('', file=fileout)
        print(']', file=fileout)


if __name__ == '__main__':
    add_defaults('gpu')
    add_argument('--adaptive_pool_height', type=int, default=16,
                 help='Average adaptive pooling of the images before the '
                      'LSTM layers')
    add_argument('--lstm_hidden_size', type=int, default=128)
    add_argument('--lstm_num_layers', type=int, default=1)
    add_argument('--add_softmax', action='store_true')
    add_argument('--add_boundary_blank', action='store_true')
    add_argument('syms', help='Symbols table mapping from strings to integers')
    add_argument('img_dir', help='Directory containing word images')
    add_argument('gt_file', help='')
    add_argument('checkpoint', help='')
    add_argument('output', type=argparse.FileType('w'))
    args = args()

    # Build neural network
    syms = SymbolsTable(args.syms)
    model = build_ctc_model(
        num_outputs=len(syms),
        adaptive_pool_height=args.adaptive_pool_height,
        lstm_hidden_size=args.lstm_hidden_size,
        lstm_num_layers=args.lstm_num_layers)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint)
    if 'model' in ckpt and 'optimizer' in ckpt:
        model.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt)

    # Ensure parameters are in the correct device
    model.eval()
    if args.gpu > 0:
        model = model.cuda(args.gpu - 1)
    else:
        model = model.cpu()

    dataset = TextImageFromTextTableDataset(
        args.gt_file, args.img_dir,
        img_transform=ImageToTensor(),
        txt_transform=TextToTensor(syms))
    dataset_loader = ImageDataLoader(dataset=dataset,
                                     image_channels=1,
                                     num_workers=8)

    with torch.cuda.device(args.gpu - 1):
        for batch in dataset_loader:
            if args.gpu > 0:
                x = batch['img'].data.cuda(args.gpu - 1)
            else:
                x = batch['img'].data.cpu()
            y = model(torch.autograd.Variable(x)).data
            if args.add_softmax:
                y = torch.nn.functional.log_softmax(y, dim=-1)
            dump_output_matrix(batch['id'], [y], args.output,
                               args.add_boundary_blank)
