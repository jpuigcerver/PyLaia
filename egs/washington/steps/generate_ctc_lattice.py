#!/usr/bin/env python

from __future__ import print_function

import torch
from dortmund_utils import build_ctc_model, DortmundImageToTensor
from laia.plugins.arguments import add_argument, add_defaults, args
from laia.utils import ImageToTensor, TextToTensor
from laia.utils.symbols_table import SymbolsTable
from laia.data import TextImageFromTextTableDataset
from laia.data import ImageDataLoader

if __name__ == '__main__':
    add_defaults('gpu')
    add_argument('--adaptive_pool_height', type=int, default=16,
                 help='Average adaptive pooling of the images before the '
                      'LSTM layers')
    add_argument('--lstm_hidden_size', type=int, default=128)
    add_argument('--lstm_num_layers', type=int, default=1)
    add_argument('syms', help='Symbols table mapping from strings to integers')
    add_argument('img_dir', help='Directory containing word images')
    add_argument('gt_file', help='')
    add_argument('checkpoint', help='')
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
    if args.gpu > 0:
        model = model.cuda(args.gpu - 1)
    else:
        model = model.cpu()

    dataset = TextImageFromTextTableDataset(
        args.va_txt_table, args.tr_img_dir,
        img_transform=ImageToTensor(),
        txt_transform=TextToTensor(syms))
    dataset_loader = ImageDataLoader(dataset=dataset,
                                     image_channels=1,
                                     batch_size=args.batch_size,
                                     num_workers=8)
    for batch in dataset_loader:
        print(batch)
        # model(batch[])
