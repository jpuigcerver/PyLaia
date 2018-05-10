#!/usr/bin/env python
from __future__ import division

import os

import numpy as np
import torch
from scipy.spatial.distance import pdist
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, desc=None):
        return x

import laia.logging as log
import laia.utils
from dortmund_utils import (build_dortmund_model, DortmundImageToTensor,
                            DortmundBCELoss)
from laia.engine.engine import EPOCH_END, EPOCH_START
from laia.engine.feeders import ImageFeeder, ItemFeeder, VariableFeeder, \
    PHOCFeeder
from laia.hooks import Hook
from laia.hooks.conditions import GEqThan
from laia.hooks.meters import AveragePrecisionMeter
from laia.plugins.arguments import add_argument, add_defaults, args


def create_dataset_and_loader(img_dir, gt_table, img_transform,
                              samples_per_epoch=None):
    def sort_key(x):
        return -x['img'].size(2)

    ds = laia.data.TextImageFromTextTableDataset(
        gt_table, img_dir, img_transform=img_transform)
    if samples_per_epoch is None:
        ds_loader = torch.utils.data.DataLoader(
            ds, batch_size=1, num_workers=8, shuffle=True,
            collate_fn=laia.data.PaddingCollater({'img': [1, None, None]},
                                                 sort_key=sort_key))
    else:
        ds_loader = torch.utils.data.DataLoader(
            ds, batch_size=1, num_workers=8,
            sampler=laia.data.FixedSizeSampler(ds, samples_per_epoch),
            collate_fn=laia.data.PaddingCollater({'img': [1, None, None]},
                                                 sort_key=sort_key))
    return ds, ds_loader


class Evaluate(object):
    def __init__(self, model, queries_loader, gpu):
        self.model = model
        self.queries_loader = queries_loader
        self.gpu = gpu

    def __call__(self, **kwargs):
        self.model.eval()
        input_fn = ImageFeeder(device=self.gpu, keep_padded_tensors=False,
                               parent_feeder=ItemFeeder('img'))
        queries_embed = []
        queries_gt = []
        for batch in tqdm(self.queries_loader, desc='Valid'):
            try:
                output = torch.nn.functional.sigmoid(
                    self.model(input_fn(batch)))
                queries_embed.append(output.data.cpu().numpy())
                queries_gt.append(batch['txt'][0])
            except Exception as ex:
                laia.logging.error('Exception processing: {!r}', batch['id'])
                raise ex
        queries_embed = np.vstack(queries_embed)

        NQ = queries_embed.shape[0]
        gap_meter = AveragePrecisionMeter(desc_sort=False)
        map_meter = [AveragePrecisionMeter(desc_sort=False) for _ in range(NQ)]
        distances = pdist(queries_embed, metric='braycurtis')
        inds = [(i, j) for i in range(NQ) for j in range(i + 1, NQ)]
        for k, (i, j) in enumerate(inds):
            if queries_gt[i] == queries_gt[j]:
                gap_meter.add(1, 0, 0, distances[k])
                map_meter[i].add(1, 0, 0, distances[k])
            else:
                gap_meter.add(0, 1, 0, distances[k])
                map_meter[i].add(0, 1, 0, distances[k])

        g_ap = gap_meter.value
        aps = [m.value for m in map_meter if m.value is not None]
        laia.logging.info('Epoch {epochs:4d}, '
                          'VA gAP = {gap:5.1%}, '
                          'VA mAP = {map:5.1%}, ',
                          epochs=kwargs['epoch'],
                          gap=g_ap,
                          map=np.mean(aps) if len(aps) > 0 else None)

if __name__ == '__main__':
    add_defaults('gpu', 'max_epochs', 'max_updates', 'train_samples_per_epoch',
                 'seed', 'train_path',
                 # Override default values for these arguments, but use the
                 # same help/checks:
                 learning_rate=0.0001,
                 momentum=0.9,
                 iterations_per_update=10,
                 show_progress_bar=True,
                 use_distortions=True,
                 weight_l2_penalty=0.00005)
    add_argument('--load_checkpoint', type=str,
                 help='Path to the checkpoint to load.')
    add_argument('--continue_epoch', type=int)
    add_argument('--phoc_levels', type=int, default=[1, 2, 3, 4, 5], nargs='+',
                 help='PHOC levels used to encode the transcript')
    add_argument('syms', help='Symbols table mapping from strings to integers')
    add_argument('img_dir', help='Directory containing word images')
    add_argument('tr_txt_table',
                 help='Character transcriptions of each training image')
    add_argument('va_txt_table',
                 help='Character transcriptions of each validation image')
    args = args()

    laia.random.manual_seed(args.seed)

    syms = laia.utils.SymbolsTable(args.syms)

    phoc_size = sum(args.phoc_levels) * len(syms)
    model = build_dortmund_model(phoc_size)
    if args.load_checkpoint:
        model_ckpt = torch.load(args.load_checkpoint)
        model.load_state_dict(model_ckpt)
    model = model.cuda(args.gpu - 1) if args.gpu > 0 else model.cpu()
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
    tr_ds, tr_ds_loader = create_dataset_and_loader(
        args.img_dir, args.tr_txt_table, tr_img_transform,
        args.train_samples_per_epoch)
    # Validation data
    ca_ds, va_ds_loader = create_dataset_and_loader(
        args.img_dir, args.va_txt_table, laia.utils.ImageToTensor())

    trainer = laia.engine.Trainer(
        model=model,
        criterion=DortmundBCELoss(),
        optimizer=optimizer,
        data_loader=tr_ds_loader,
        batch_input_fn=ImageFeeder(
            device=args.gpu,
            keep_padded_tensors=False,
            parent_feeder=ItemFeeder('img')),
        batch_target_fn=VariableFeeder(
            device=args.gpu,
            parent_feeder=PHOCFeeder(
                syms=syms,
                levels=args.phoc_levels,
                parent_feeder=ItemFeeder('txt'))),
        progress_bar='Train' if args.show_progress_bar else False)
    trainer.iterations_per_update = args.iterations_per_update

    trainer.add_hook(EPOCH_END, Evaluate(model, va_ds_loader, args.gpu))

    if args.max_epochs and args.max_epochs > 0:
        trainer.add_hook(EPOCH_START,
                         Hook(GEqThan(trainer.epochs, args.max_epochs),
                              trainer.stop))

    if args.continue_epoch:
        trainer._epochs = args.continue_epoch

    # Launch training
    trainer.run()

    # Save model parameters after training
    torch.save(model.state_dict(), os.path.join(args.train_path, 'model.ckpt'))
