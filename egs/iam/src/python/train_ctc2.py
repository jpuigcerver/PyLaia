#!/usr/bin/env python

from __future__ import absolute_import

import argparse
import multiprocessing as mp
import os

import laia.common.logging as log
from laia.data import (ImageDataLoader, TextImageFromTextTableDataset,
                       FixedSizeSampler)
from laia.engine import Trainer, Evaluator, HtrEngineWrapper
from laia.engine.engine import EPOCH_END, EPOCH_START
from laia.engine.feeders import ImageFeeder, ItemFeeder
from laia.hooks import Hook, HookList, action, Action, ActionList
from laia.hooks.conditions import Lowest, MultipleOf, GEqThan
from laia.common.arguments import add_argument, args, add_defaults
from laia.common.loader import (TrainerLoader, ModelLoader,
                                TrainerCheckpointLoader, ModelCheckpointLoader)
from laia.common.saver import (TrainerSaver, CheckpointSaver, RollingSaver,
                               ModelCheckpointSaver, TrainerCheckpointSaver)
from laia.utils import SymbolsTable, ImageToTensor, TextToTensor
from laia.utils.dortmund_image_to_tensor import DortmundImageToTensor
from torch.optim import SGD

if __name__ == '__main__':
    add_defaults('batch_size', 'learning_rate', 'gpu',
                 'max_epochs', 'train_path',
                 'train_samples_per_epoch', 'iterations_per_update',
                 momentum=0.9,
                 show_progress_bar=True)
    add_argument('fixed_height', type=int,
                 help='Resize images to this fixed height size')
    add_argument('syms', type=argparse.FileType('r'),
                 help='Symbols table mapping from strings to integers')
    add_argument('img_dir',
                 help='Directory containing word images')
    add_argument('tr_txt_table', type=argparse.FileType('r'),
                 help='Character transcriptions of each training image')
    add_argument('va_txt_table', type=argparse.FileType('r'),
                 help='Character transcriptions of each validation image')
    args = args()

    syms = SymbolsTable(args.syms)

    model = ModelLoader(args.train_path, gpu=args.gpu).load()
    if model is None:
        log.error('Could not find the model. Have you run '
                  '"pylaia-htr-create-model"?')
        exit(1)
    model = model.cuda(args.gpu - 1) if args.gpu else model.cpu()
    log.info('Model has {} parameters', sum(param.data.numel()
                                            for param in model.parameters()))

    trainer = TrainerLoader(args.train_path, gpu=args.gpu).load()
    if trainer is None:
        optimizer = SGD(model.parameters(),
                        lr=args.learning_rate,
                        momentum=args.momentum)
        parameters = {
            'model': model,
            'criterion': None,  # Set automatically by HtrEngineWrapper
            'optimizer': optimizer,
            'batch_target_fn': ItemFeeder('txt'),
            'progress_bar': 'Train' if args.show_progress_bar else None}
        trainer = Trainer(**parameters)
        TrainerSaver(args.train_path).save(Trainer, **parameters)

    tr_ds = TextImageFromTextTableDataset(
        args.tr_txt_table, args.img_dir,
        img_transform=DortmundImageToTensor(fixed_height=args.fixed_height),
        txt_transform=TextToTensor(syms))

    tr_ds_loader = ImageDataLoader(dataset=tr_ds,
                                   image_channels=1,
                                   batch_size=args.batch_size,
                                   num_workers=mp.cpu_count(),
                                   shuffle=not bool(
                                       args.train_samples_per_epoch),
                                   sampler=FixedSizeSampler(
                                       tr_ds,
                                       args.train_samples_per_epoch)
                                   if args.train_samples_per_epoch else None)

    # Set all these separately because they might change between executions
    trainer.iterations_per_update = args.iterations_per_update
    trainer.set_data_loader(tr_ds_loader)
    trainer.set_batch_input_fn(ImageFeeder(device=args.gpu,
                                           # keep_padded_tensors=False,
                                           parent_feeder=ItemFeeder('img')))

    va_ds = TextImageFromTextTableDataset(
        args.va_txt_table, args.img_dir,
        img_transform=ImageToTensor(fixed_height=args.fixed_height),
        txt_transform=TextToTensor(syms))
    va_ds_loader = ImageDataLoader(dataset=va_ds,
                                   image_channels=1,
                                   batch_size=args.batch_size,
                                   num_workers=mp.cpu_count())

    evaluator = Evaluator(
        model=model,
        data_loader=va_ds_loader,
        batch_input_fn=ImageFeeder(device=args.gpu,
                                   parent_feeder=ItemFeeder('img')),
        batch_target_fn=ItemFeeder('txt'),
        progress_bar='Valid' if args.show_progress_bar else None)

    engine_wrapper = HtrEngineWrapper(trainer, evaluator)
    engine_wrapper.set_word_delimiters([])


    def ckpt_saver(filename):
        return CheckpointSaver(os.path.join(args.train_path, filename))


    tr_saver_best_cer = RollingSaver(
        TrainerCheckpointSaver(
            ckpt_saver('engine-wrapper.ckpt.lowest-valid-cer'),
            engine_wrapper, gpu=args.gpu), keep=2)
    tr_saver_best_wer = RollingSaver(
        TrainerCheckpointSaver(
            ckpt_saver('engine-wrapper.ckpt.lowest-valid-wer'),
            engine_wrapper, gpu=args.gpu), keep=2)
    tr_saver = RollingSaver(
        TrainerCheckpointSaver(ckpt_saver('engine-wrapper.ckpt'),
                               engine_wrapper, gpu=args.gpu))
    mo_saver_best_cer = RollingSaver(
        ModelCheckpointSaver(ckpt_saver('model.ckpt.lowest-valid-cer'),
                             model), keep=2)
    mo_saver_best_wer = RollingSaver(
        ModelCheckpointSaver(ckpt_saver('model.ckpt.lowest-valid-wer'),
                             model), keep=2)
    mo_saver = RollingSaver(
        ModelCheckpointSaver(ckpt_saver('model.ckpt'), model))


    @action
    def save(saver, epoch):
        saver.save(suffix=epoch)


    log.get_logger('laia.hooks.conditions.multiple_of').setLevel(log.WARNING)

    # Set hooks
    trainer.add_hook(EPOCH_END, HookList(
        # Save on best CER
        Hook(Lowest(engine_wrapper.valid_cer(), name='Lowest CER'),
             ActionList(Action(save, saver=tr_saver_best_cer),
                        Action(save, saver=mo_saver_best_cer))),
        # Save on best WER
        Hook(Lowest(engine_wrapper.valid_wer(), name='Lowest WER'),
             ActionList(Action(save, saver=tr_saver_best_wer),
                        Action(save, saver=mo_saver_best_wer))),
        # Save every 5 epochs
        Hook(MultipleOf(trainer.epochs, 5),
             ActionList(Action(save, saver=tr_saver),
                        Action(save, saver=mo_saver)))))

    if args.max_epochs:
        # Save always the last 5 epochs
        trainer.add_hook(
            EPOCH_END,
            Hook(GEqThan(trainer.epochs, args.max_epochs - 5),
                 ActionList(Action(save, saver=tr_saver),
                            Action(save, saver=mo_saver))))
        # Stop if the number of epochs has been reached
        trainer.add_hook(
            EPOCH_START,
            Hook(GEqThan(trainer.epochs, args.max_epochs),
                 trainer.stop))

    # Continue from last checkpoints with lowest valid cer, if possible
    TrainerCheckpointLoader(engine_wrapper, gpu=args.gpu).load_by(
        os.path.join(args.train_path, 'engine-wrapper.ckpt.lowest-valid-cer*'))
    ModelCheckpointLoader(model, gpu=args.gpu).load_by(
        os.path.join(args.train_path, 'model.ckpt.lowest-valid-cer*'))

    engine_wrapper.run()
