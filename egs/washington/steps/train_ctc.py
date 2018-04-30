#!/usr/bin/env python

from __future__ import division

import os

import torch

import laia.utils
from dortmund_utils import (build_ctc_model, build_ctc_model2,
                            DortmundImageToTensor, ModelCheckpointKeepLastSaver)
from laia.engine.engine import EPOCH_START, EPOCH_END
from laia.engine.feeders import ImageFeeder, ItemFeeder
from laia.engine.htr_engine_wrapper import HtrEngineWrapper
from laia.engine.trainer import Trainer
from laia.hooks import Hook, HookCollection
from laia.hooks.conditions import Lowest, GEqThan
from laia.plugins.arguments import add_argument, add_defaults, args


logger = laia.logging.get_logger('laia.egs.washington.train_ctc')

if __name__ == '__main__':
    add_defaults('gpu', 'max_epochs', 'max_updates', 'train_samples_per_epoch',
                 'valid_samples_per_epoch', 'seed', 'save_path',
                 # Override default values for these arguments, but use the
                 # same help/checks:
                 learning_rate=0.0001,
                 momentum=0.9,
                 iterations_per_update=10,
                 show_progress_bar=True,
                 use_distortions=True,
                 weight_l2_penalty=0.00005)
    add_argument('--train_laia', action='store_true',
                 help='Train Laia-based model')
    add_argument('--adaptive_pool_height', type=int, default=16,
                 help='Average adaptive pooling of the images before the '
                      'LSTM layers')
    add_argument('--cnn_num_filters', type=int, nargs='+',
                 default=[16, 32, 48, 64])
    add_argument('--cnn_maxpool_size', type=int, nargs='*', default=[2, 2])
    add_argument('--lstm_hidden_size', type=int, default=128)
    add_argument('--lstm_num_layers', type=int, default=1)
    add_argument('syms', help='Symbols table mapping from strings to integers')
    add_argument('tr_img_dir', help='Directory containing word images')
    add_argument('tr_txt_table',
                 help='Character transcriptions of each training image')
    add_argument('va_txt_table',
                 help='Character transcriptions of each validation image')
    args = args()
    laia.random.manual_seed(args.seed)

    syms = laia.utils.SymbolsTable(args.syms)

    # If --use_distortions is given, apply the same affine distortions used by
    # Dortmund University.
    if args.use_distortions:
        tr_img_transform = DortmundImageToTensor()
    else:
        tr_img_transform = laia.utils.ImageToTensor()

    # Training data
    tr_ds = laia.data.TextImageFromTextTableDataset(
        args.tr_txt_table, args.tr_img_dir,
        img_transform=tr_img_transform,
        txt_transform=laia.utils.TextToTensor(syms))
    if args.train_samples_per_epoch is None:
        tr_ds_loader = laia.data.ImageDataLoader(
            tr_ds, image_channels=1, batch_size=1, num_workers=8, shuffle=True)
    else:
        tr_ds_loader = laia.data.ImageDataLoader(
            tr_ds, image_channels=1, batch_size=1, num_workers=8,
            sampler=laia.data.FixedSizeSampler(tr_ds,
                                               args.train_samples_per_epoch))

    # Validation data
    va_ds = laia.data.TextImageFromTextTableDataset(
        args.va_txt_table, args.tr_img_dir,
        img_transform=laia.utils.ImageToTensor(),
        txt_transform=laia.utils.TextToTensor(syms))
    if args.valid_samples_per_epoch is None:
        va_ds_loader = laia.data.ImageDataLoader(
            va_ds, image_channels=1, batch_size=1, num_workers=8, shuffle=True)
    else:
        va_ds_loader = laia.data.ImageDataLoader(
            va_ds, image_channels=1, batch_size=1, num_workers=8,
            sampler=laia.data.FixedSizeSampler(va_ds,
                                               args.valid_samples_per_epoch))


    if args.train_laia:
        model = build_ctc_model2(
            cnn_num_filters=args.cnn_num_filters,
            cnn_maxpool_size=args.cnn_maxpool_size,
            adaptive_pool_height=args.adaptive_pool_height,
            lstm_hidden_size=args.lstm_hidden_size,
            lstm_num_layers=args.lstm_num_layers,
            num_outputs=len(syms))
    else:
        model = build_ctc_model(
            adaptive_pool_height=args.adaptive_pool_height,
            lstm_hidden_size=args.lstm_hidden_size,
            lstm_num_layers=args.lstm_num_layers,
            num_outputs=len(syms))
    model = model.cuda(args.gpu - 1) if args.gpu > 0 else model.cpu()
    logger.info('Model has {} parameters',
                sum(param.data.numel() for param in model.parameters()))

    optimizer = torch.optim.SGD(params=model.parameters(),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_l2_penalty)
    parameters = {
        'model': model,
        'criterion': None,  # Set automatically by HtrEngineWrapper
        'optimizer': optimizer,
        'data_loader': tr_ds_loader,
        'batch_input_fn': ImageFeeder(device=args.gpu,
                                      keep_padded_tensors=False,
                                      parent_feeder=ItemFeeder('img')),
        'batch_target_fn': ItemFeeder('txt'),
        'progress_bar': 'Train' if args.show_progress_bar else False,
    }
    trainer = Trainer(**parameters)
    trainer.iterations_per_update = args.iterations_per_update

    evaluator = laia.engine.Evaluator(
        model=model,
        data_loader=va_ds_loader,
        batch_input_fn=ImageFeeder(device=args.gpu,
                                   keep_padded_tensors=False,
                                   parent_feeder=ItemFeeder('img')),
        batch_target_fn=ItemFeeder('txt'),
        progress_bar='Valid' if args.show_progress_bar else False)

    engine_wrapper = HtrEngineWrapper(trainer, evaluator)
    engine_wrapper.set_word_delimiters([])

    lowest_cer_saver = ModelCheckpointKeepLastSaver(
        model,
        os.path.join(args.save_path, 'model.ckpt-lowest-valid-cer'))
    lowest_wer_saver = ModelCheckpointKeepLastSaver(
        model,
        os.path.join(args.save_path, 'model.ckpt-lowest-valid-wer'))

    # Set hooks
    trainer.add_hook(EPOCH_END, HookCollection(
        Hook(Lowest(engine_wrapper.valid_cer(), name='Lowest CER'),
             lowest_cer_saver),
        Hook(Lowest(engine_wrapper.valid_wer(), name='Lowest WER'),
             lowest_wer_saver)))
    if args.max_epochs and args.max_epochs > 0:
        trainer.add_hook(EPOCH_START,
                         Hook(GEqThan(trainer.epochs, args.max_epochs),
                              trainer.stop))

    # Launch training
    engine_wrapper.run()

    # Save model parameters after training
    torch.save(model.state_dict(), os.path.join(args.save_path, 'model.ckpt'))
