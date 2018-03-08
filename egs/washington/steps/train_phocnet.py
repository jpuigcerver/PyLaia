#!/usr/bin/env python
from __future__ import division

import logging

import torch

import laia.utils
from dortmund_utils import build_dortmund_model, DortmundImageToTensor
from laia.engine.phoc_engine_wrapper import PHOCEngineWrapper
from laia.engine.triggers import Any, MaxEpochs
from laia.utils.arguments import add_argument, add_defaults, args

if __name__ == '__main__':
    logging.basicConfig()

    add_defaults('gpu', 'max_epochs', 'num_samples_per_epoch', 'seed',
                 'train_loss_std_window_size', 'train_loss_std_threshold',
                 'valid_map_std_window_size', 'valid_map_std_threshold',
                 # Override default values for these arguments, but use the
                 # same help/checks:
                 batch_size=1,
                 learning_rate=0.0001,
                 momentum=0.9,
                 num_iterations_per_update=10,
                 show_progress_bar=True,
                 use_distortions=True,
                 weight_l2_penalty=0.00005)

    add_argument('--phoc_levels', type=int, default=[1, 2, 3, 4, 5], nargs='+',
                 help='PHOC levels used to encode the transcript')
    add_argument('syms', help='Symbols table mapping from strings to integers')
    add_argument('tr_img_dir', help='Directory containing word images')
    add_argument('tr_txt_table',
                 help='Character transcriptions of each training image')
    add_argument('va_txt_table',
                 help='Character transcriptions of each validation image')
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
                                weight_decay=args.weight_l2_penalty)

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
            sampler=laia.data.FixedSizeSampler(tr_ds,
                                               args.num_samples_per_epoch),
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
    if args.train_loss_std_window_size and args.train_loss_std_threshold:
        early_stop_triggers.append(
            laia.engine.triggers.MeterStandardDeviation(
                meter=engine_wrapper.train_loss,
                threshold=args.train_loss_std_threshold,
                num_values_to_keep=args.train_loss_std_window_size,
                meter_key=0))

    # Configure trigger on the validation map
    if args.valid_map_std_window_size and args.valid_map_std_threshold:
        early_stop_triggers.append(
            laia.engine.triggers.MeterStandardDeviation(
                meter=engine_wrapper.valid_ap,
                threshold=args.valid_map_std_threshold,
                num_values_to_keep=args.valid_map_std_window_size,
                meter_key=1))

    trainer.set_early_stop_trigger(Any(*early_stop_triggers))
    trainer.set_num_iterations_per_update(args.num_iterations_per_update)

    # Launch training
    engine_wrapper.run()
