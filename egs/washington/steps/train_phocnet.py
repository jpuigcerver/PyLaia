#!/usr/bin/env python
from __future__ import division

import os

import torch

import laia.logging as log
import laia.utils
from dortmund_utils import build_dortmund_model, DortmundImageToTensor
from laia.engine.phoc_engine_wrapper import PHOCEngineWrapper
from laia.engine.triggers import Any, NumEpochs
from laia.plugins.arguments import add_argument, add_defaults, args


class SaveModelCheckpointHook(object):
    def __init__(self, meter, filename, title, key):
        self._meter = meter
        self._highest = -float('inf')
        self._filename = os.path.normpath(filename)
        self._title = title
        self._key = key
        dirname = os.path.dirname(self._filename)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    def _save(self, model_dict):
        torch.save(model_dict, self._filename)

    def __call__(self, caller, **_):
        assert isinstance(caller, laia.engine.Engine)
        if self._meter.value[self._key] > self._highest:
            self._highest = self._meter.value[self._key]
            laia.logging.get_logger().info(
                'New highest {}: {:5.1%}'.format(self._title, self._highest))
            self._save(caller.model.state_dict())


if __name__ == '__main__':
    add_defaults('gpu', 'max_epochs', 'max_updates', 'num_samples_per_epoch',
                 'seed',
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
    add_argument('--model_checkpoint', type=str, default='model.ckpt',
                 help='Filename of the output model checkpoint')
    add_argument('syms', help='Symbols table mapping from strings to integers')
    add_argument('tr_img_dir', help='Directory containing word images')
    add_argument('tr_txt_table',
                 help='Character transcriptions of each training image')
    add_argument('va_txt_table',
                 help='Character transcriptions of each validation image')
    args = args()

    laia.random.manual_seed(args.seed)

    syms = laia.utils.SymbolsTable(args.syms)

    phoc_size = sum(args.phoc_levels) * len(syms)
    model = build_dortmund_model(phoc_size)
    if args.gpu > 0:
        model = model.cuda(args.gpu - 1)
    else:
        model = model.cpu()

    log.info('Model has {} parameters',
             sum(param.data.numel() for param in model.parameters()))
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

    # List of early stop triggers.
    # If any of these returns True, training will stop.
    early_stop_triggers = []

    # Configure NumEpochs trigger
    if args.max_epochs and args.max_epochs > 0:
        early_stop_triggers.append(
            NumEpochs(trainer=trainer, num_epochs=args.max_epochs,
                      name='Max training epochs'))

    # Configure NumUpdates trigger
    # TODO(jpuigcerver): Trainer needs to evaluate early stop in other places
    """
    if args.max_updates and args.max_updates > 0:
        early_stop_triggers.append(
            NumUpdates(trainer=trainer, num_updates=args.max_updates,
                       name='Max training updates'))
    """

    # Configure trigger on the training loss
    if args.train_loss_std_window_size and args.train_loss_std_threshold:
        early_stop_triggers.append(
            laia.engine.triggers.MeterStandardDeviation(
                meter=engine_wrapper.train_loss,
                meter_key=0,
                num_values_to_keep=args.train_loss_std_window_size,
                threshold=args.train_loss_std_threshold,
                name='Train loss standard deviation'))

    # Configure trigger on the validation map
    if args.valid_map_std_window_size and args.valid_map_std_threshold:
        early_stop_triggers.append(
            laia.engine.triggers.MeterStandardDeviation(
                meter=engine_wrapper.valid_ap,
                meter_key=1,
                num_values_to_keep=args.valid_map_std_window_size,
                threshold=args.valid_map_std_threshold,
                name='Valid mAP standard deviation'))

    trainer.set_early_stop_trigger(Any(*early_stop_triggers))
    trainer.set_num_iterations_per_update(args.num_iterations_per_update)

    if args.model_checkpoint:
        filename_gap = args.model_checkpoint + '-valid-highest-gap'
        evaluator.add_hook(
            evaluator.ON_EPOCH_END,
            SaveModelCheckpointHook(engine_wrapper.valid_ap, filename_gap,
                                    title='gAP', key=0))

        filename_map = args.model_checkpoint + '-valid-highest-map'
        evaluator.add_hook(
            evaluator.ON_EPOCH_END,
            SaveModelCheckpointHook(engine_wrapper.valid_ap, filename_map,
                                    title='mAP', key=1))

    # Launch training
    engine_wrapper.run()

    # Save model parameters after training
    torch.save(model.state_dict(), args.model_checkpoint)
