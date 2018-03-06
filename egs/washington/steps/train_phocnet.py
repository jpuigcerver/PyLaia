#!/usr/bin/env python
from __future__ import division

import logging
import torch
import laia.utils

from laia.engine.triggers import Any, MaxEpochs
from laia.engine.phoc_engine_wrapper import PHOCEngineWrapper
from laia.utils.arguments import add_argument, add_defaults, args, str2bool
from dortmund_utils import build_dortmund_model, DortmundBCELoss, DortmundImageToTensor


if __name__ == '__main__':
    logging.basicConfig()

    add_defaults('gpu', 'seed', 'max_epochs')
    add_argument('--batch_size', type=int, default=1,
                 help='Batch size')
    add_argument('--learning_rate', type=float, default=0.0001,
                 help='Learning rate')
    add_argument('--momentum', type=float, default=0.9,
                 help='Momentum')
    add_argument('--num_iterations_to_update', type=int, default=10,
                 help='Update parameters every n iterations')
    add_argument('--num_samples_per_epoch', type=int, default=None,
                 help='Use this number of samples randomly sampled '
                      'from the dataset in each epoch')
    add_argument('--phoc_levels', type=int, default=[1, 2, 3, 4, 5], nargs='+',
                 help='PHOC levels used to encode the transcript')
    add_argument('--show_progress_bar', type=bool, default=True,
                 help='If true, show progress bar for each epoch')
    add_argument('--use_distortions', type=str2bool, nargs='?', const=True, default=True,
                 help='Use dynamic distortions to augment the training data')
    add_argument('--weight_decay', type=float, default=0.00005,
                 help='L2 weight decay')
    add_argument('--train_loss_stddev_window_size', type=int, default=None,
                 help='Use this number of epochs to compute the std. dev. of '
                      'the training loss (must be >= 2)')
    add_argument('--train_loss_stddev_threshold', type=float, default=0.1,
                 help='Stop training when the std. dev. of the training loss '
                      'is below this threshold')
    add_argument('syms')
    add_argument('tr_img_dir')
    add_argument('tr_txt_table')
    add_argument('va_txt_table')
    args = args()

    laia.manual_seed(args.seed)

    syms = laia.utils.SymbolsTable(args.syms)

    phoc_size = sum(args.phoc_levels) * len(syms)
    model = build_dortmund_model(phoc_size)
    if args.gpu > 0:
        model = model.cuda(args.gpu - 1)
    else:
        model = model.cpu()

    logging.info('Model has {} parameters'.format(
        sum(param.data.numel() for param in model.parameters())))
    optimizer = torch.optim.SGD(params=model.parameters(),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # If --use_distortions is given, apply the same affine distortions used by
    # Dortmund University.
    if args.use_distortions:
        tr_img_transform = DortmundImageToTensor()
    else:
        tr_img_transform = laia.utils.ImageToTensor()

    # Training data
    tr_ds = laia.data.TextImageFromTextTableDataset(
        args.tr_txt_table, args.tr_img_dir, img_transform=tr_img_transform)
    if args.num_samples_per_epoch is None:
        tr_ds_loader = torch.utils.data.DataLoader(
            tr_ds, batch_size=args.batch_size, num_workers=8, shuffle=True,
            collate_fn=laia.data.PaddingCollater({
                'img': [1, None, None],
            }, sort_key=lambda x: -x['img'].size(2)))
    else:
        tr_ds_loader = torch.utils.data.DataLoader(
            tr_ds, batch_size=args.batch_size, num_workers=8,
            sampler=laia.data.FixedSizeSampler(tr_ds, args.num_samples_per_epoch),
            collate_fn=laia.data.PaddingCollater({
                'img': [1, None, None],
            }, sort_key=lambda x: -x['img'].size(2)))

    # Validation data
    va_ds = laia.data.TextImageFromTextTableDataset(
        args.va_txt_table, args.tr_img_dir,
        img_transform=laia.utils.ImageToTensor())
    va_ds_loader = torch.utils.data.DataLoader(
        va_ds, args.batch_size, num_workers=8,
        collate_fn=laia.data.PaddingCollater({
            'img': [1, None, None],
        }, sort_key=lambda x: -x['img'].size(2)))

    trainer = laia.engine.Trainer(
        model=model,
        # Note: Criterion will be set automatically by the wrapper
        criterion=None,
        optimizer=optimizer,
        data_loader=tr_ds_loader,
        progress_bar='Train' if args.show_progress_bar else False)

    evaluator = laia.engine.Evaluator(
        model=model,
        data_loader=va_ds_loader,
        progress_bar='Valid' if args.show_progress_bar else False)

    engine_wrapper = PHOCEngineWrapper(
        symbols_table=syms,
        phoc_levels=args.phoc_levels,
        train_engine=trainer,
        valid_engine=evaluator,
        gpu=args.gpu)
    engine_wrapper.logger.setLevel(logging.INFO)

    # List of early stop triggers.
    # If any of these returns True, training will stop.
    early_stop_triggers = []

    # Configure MaxEpochs trigger
    if args.max_epochs and args.max_epochs > 0:
        early_stop_triggers.append(
            MaxEpochs(trainer=trainer, max_epochs=args.max_epochs))

    # Configure trigger on the training loss
    if (args.train_loss_stddev_window_size and
            args.train_loss_stddev_window_size > 1):
        early_stop_triggers.append(
            laia.engine.triggers.MeterStandardDeviation(
                meter=engine_wrapper.train_loss,
                threshold=args.train_loss_stddev_threshold,
                num_values_to_keep=args.train_loss_stddev_window_size))

    trainer.set_early_stop_trigger(Any(*early_stop_triggers))

    # Launch training
    engine_wrapper.run()
