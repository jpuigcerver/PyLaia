#!/usr/bin/env python

from __future__ import print_function

import argparse

import torch
from dortmund_utils import build_dortmund_model
from laia.data import ImageDataLoader
from laia.data import TextImageFromTextTableDataset
from laia.plugins.arguments import add_argument, add_defaults, args
from laia.utils import ImageToTensor
from laia.utils.symbols_table import SymbolsTable
from tqdm import tqdm
import math


def phoc_lattice(img_ids, outputs, fileout):
    for img_id, output in zip(img_ids, outputs):
        output = output.cpu()
        print(img_id, file=fileout)
        for t in range(output.size(0)):
            lp1 = float(output[t])
            p0 = -math.expm1(lp1)
            lp0 = math.log(p0) if p0 > 0 else float('-inf')
            for k, p in enumerate([lp0, lp1], 1):
                if not math.isinf(p):
                    print('{:d}\t{:d}\t{:d}\t0,{:.10g},{:d}'.format(
                        t, t + 1, k, -float(p), k), file=fileout)
        print(output.size(0), file=fileout)
        print('', file=fileout)


if __name__ == '__main__':
    add_defaults('gpu')
    add_argument('--phoc_levels', type=int, default=[1, 2, 3, 4, 5], nargs='+',
                 help='PHOC levels used to encode the transcript')
    add_argument('--add_sigmoid', action='store_true')
    add_argument('syms', help='Symbols table mapping from strings to integers')
    add_argument('img_dir', help='Directory containing word images')
    add_argument('gt_file', help='')
    add_argument('checkpoint', help='')
    add_argument('output', type=argparse.FileType('w'))
    args = args()

    # Build neural network
    syms = SymbolsTable(args.syms)
    phoc_size = sum(args.phoc_levels) * len(syms)
    model = build_dortmund_model(phoc_size)

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
        img_transform=ImageToTensor())
    dataset_loader = ImageDataLoader(dataset=dataset,
                                     image_channels=1,
                                     num_workers=8)

    with torch.cuda.device(args.gpu - 1):
        for batch in tqdm(dataset_loader):
            if args.gpu > 0:
                x = batch['img'].data.cuda(args.gpu - 1)
            else:
                x = batch['img'].data.cpu()
            y = model(torch.autograd.Variable(x))
            if args.add_sigmoid:
                y = torch.nn.functional.logsigmoid(y)
            phoc_lattice(batch['id'], y, args.output)
