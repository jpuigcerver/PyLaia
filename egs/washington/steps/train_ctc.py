#!/usr/bin/env python

from __future__ import division

import os

import laia.data
import laia.engine
import laia.logging
import laia.nn
import laia.utils
import torch
from dortmund_utils import build_ctc_model, DortmundImageToTensor
from laia.engine.feeders import ImageFeeder, ItemFeeder
from laia.engine.triggers import (Any, NumEpochs,
                                  MeterStandardDeviation)
from laia.plugins.arguments import add_argument, add_defaults, args

logger = laia.logging.get_logger('laia.egs.washington.train_ctc')


class SaveModelCheckpointHook(object):
    def __init__(self, meter, filename):
        self._meter = meter
        self._lowest = float('inf')
        self._filename = os.path.normpath(filename)
        dirname = os.path.dirname(self._filename)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname)

    def _save(self, model_state_dict, optimizer_state_dict):
        torch.save({
            'model': model_state_dict,
            'optimizer': optimizer_state_dict
        }, self._filename)

    def __call__(self, caller, **_):
        assert isinstance(caller, laia.engine.Trainer)
        if self._meter.value < self._lowest:
            self._lowest = self._meter.value
            logger.info('New lowest validation CER: {:5.1%}', self._lowest)
            self._save(caller.model.state_dict(),
                       caller.optimizer.state_dict())


if __name__ == '__main__':
    add_defaults('gpu', 'max_epochs', 'max_updates', 'num_samples_per_epoch',
                 'seed',
                 'train_loss_std_window_size', 'train_loss_std_threshold',
                 'valid_cer_std_window_size', 'valid_cer_std_threshold',
                 # Override default values for these arguments, but use the
                 # same help/checks:
                 batch_size=1,
                 learning_rate=0.0005,
                 momentum=0.9,
                 num_iterations_per_update=10,
                 show_progress_bar=True,
                 use_distortions=True,
                 weight_l2_penalty=0.00005)
    add_argument('--save_checkpoint', type=str, default='model.ckpt',
                 help='Filename of the output model checkpoint')
    add_argument('--load_checkpoint', type=str, default=None,
                 help='Load model parameters from this checkpoint')
    add_argument('--adaptive_pool_height', type=int, default=16,
                 help='Average adaptive pooling of the images before the '
                      'LSTM layers')
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
    model = build_ctc_model(num_outputs=len(syms),
                            adaptive_pool_height=args.adaptive_pool_height,
                            lstm_hidden_size=args.lstm_hidden_size,
                            lstm_num_layers=args.lstm_num_layers)
    if args.gpu > 0:
        model = model.cuda(args.gpu - 1)
    else:
        model = model.cpu()

    optimizer = torch.optim.SGD(params=model.parameters(),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_l2_penalty)

    if args.load_checkpoint:
        logger.info('Loading parameters from {!r}', args.load_checkpoint)
        saved_dict = torch.load(args.load_checkpoint)
        if 'model' in saved_dict and 'optimizer' in saved_dict:
            model.load_state_dict(saved_dict['model'])
            optimizer.load_state_dict(saved_dict['optimizer'])
        else:
            model.load_state_dict(saved_dict)

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
    if args.num_samples_per_epoch is None:
        tr_ds_loader = laia.data.ImageDataLoader(
            tr_ds, image_channels=1, batch_size=1, num_workers=8, shuffle=True)
    else:
        tr_ds_loader = laia.data.ImageDataLoader(
            tr_ds, image_channels=1, batch_size=1, num_workers=8,
            sampler=laia.data.FixedSizeSampler(tr_ds,
                                               args.num_samples_per_epoch))

    # Validation data
    va_ds = laia.data.TextImageFromTextTableDataset(
        args.va_txt_table, args.tr_img_dir,
        img_transform=laia.utils.ImageToTensor(),
        txt_transform=laia.utils.TextToTensor(syms))
    va_ds_loader = laia.data.ImageDataLoader(dataset=va_ds,
                                             image_channels=1,
                                             batch_size=args.batch_size,
                                             num_workers=8)

    trainer = laia.engine.Trainer(
        model=model,
        criterion=None,
        optimizer=optimizer,
        data_loader=tr_ds_loader,
        batch_input_fn=ImageFeeder(device=args.gpu,
                                   keep_padded_tensors=False,
                                   parent_feeder=ItemFeeder('img')),
        batch_target_fn=ItemFeeder('txt'),
        progress_bar='Train' if args.show_progress_bar else False)

    evaluator = laia.engine.Evaluator(
        model=model,
        data_loader=va_ds_loader,
        batch_input_fn=ImageFeeder(device=args.gpu,
                                   keep_padded_tensors=False,
                                   parent_feeder=ItemFeeder('img')),
        batch_target_fn=ItemFeeder('txt'),
        progress_bar='Valid' if args.show_progress_bar else False)

    engine_wrapper = laia.engine.HtrEngineWrapper(trainer, evaluator)

    # List of early stop triggers.
    # If any of these returns True, training will stop.
    early_stop_triggers = []

    # Configure NumEpochs trigger
    if args.max_epochs and args.max_epochs > 0:
        early_stop_triggers.append(
            NumEpochs(trainer=trainer, num_epochs=args.max_epochs))

    # Configure MeterStandardDeviation trigger to monitor validation CER
    if args.valid_cer_std_window_size and args.valid_cer_std_threshold:
        early_stop_triggers.append(
            MeterStandardDeviation(
                meter=engine_wrapper.valid_cer,
                threshold=args.valid_cer_std_threshold,
                num_values_to_keep=args.valid_cer_std_window_size))

    trainer.set_early_stop_trigger(Any(*early_stop_triggers))
    trainer.set_num_iterations_per_update(args.num_iterations_per_update)

    if args.save_checkpoint:
        filename_va = args.save_checkpoint + '-valid-lowest-cer'
        trainer.add_hook(
            trainer.ON_EPOCH_END,
            SaveModelCheckpointHook(engine_wrapper.valid_cer, filename_va))

    # Start training
    with torch.cuda.device(args.gpu - 1):
        engine_wrapper.run()

    # Save model parameters after training
    if args.save_checkpoint:
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, args.save_checkpoint)
