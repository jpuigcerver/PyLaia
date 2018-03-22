#!/usr/bin/env python

from __future__ import division

import torch
from dortmund_utils import build_conv_model, DortmundImageToTensor
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence

import laia.data
import laia.engine
import laia.nn
import laia.utils
from dortmund_utils import build_conv_model, DortmundImageToTensor
from laia.engine.triggers import (Any, NumEpochs,
                                  MeterStandardDeviation)
from laia.plugins.arguments import add_argument, add_defaults, args
from laia.plugins import SaverTrigger, SaverTriggerCollection


def build_model(num_outputs,
                adaptive_pool_height=2,
                lstm_hidden_size=128,
                lstm_num_layers=1):
    class RNNWrapper(torch.nn.Module):
        def __init__(self, module):
            super(RNNWrapper, self).__init__()
            self._module = module

        def forward(self, input):
            if isinstance(input, tuple):
                input = input[0]
            return self._module(input)

        def __repr__(self):
            return repr(self._module)

    class PackedSequenceWrapper(torch.nn.Module):
        def __init__(self, module):
            super(PackedSequenceWrapper, self).__init__()
            self._module = module

        def forward(self, input):
            if isinstance(input, PackedSequence):
                x, xs = input.data, input.batch_sizes
            else:
                x, xs = input, None

            y = self._module(x)
            print(y.size())
            if xs is None:
                return pack_padded_sequence(input=y, lengths=[y.size(0)])
            else:
                return PackedSequence(data=y, batch_sizes=xs)

        def __repr__(self):
            return repr(self._module)

    m = build_conv_model()
    m.add_module('adap_pool', laia.nn.AdaptiveAvgPool2d(
        output_size=(adaptive_pool_height, None)))
    m.add_module('collapse', laia.nn.ImageToSequence(return_packed=True))
    m.add_module('blstm', torch.nn.LSTM(
        input_size=512 * adaptive_pool_height,
        hidden_size=lstm_hidden_size,
        num_layers=lstm_num_layers,
        dropout=0.5,
        bidirectional=True))
    m.add_module('linear',
                 RNNWrapper(PackedSequenceWrapper(
                     torch.nn.Linear(2 * lstm_hidden_size, num_outputs))))
    return m


if __name__ == '__main__':
    add_defaults('gpu', 'max_epochs', 'max_updates', 'num_samples_per_epoch',
                 'seed',
                 'train_loss_std_window_size', 'train_loss_std_threshold',
                 'valid_cer_std_window_size', 'valid_cer_std_threshold',
                 # Override default values for these arguments, but use the
                 # same help/checks:
                 batch_size=1,
                 learning_rate=0.0001,
                 momentum=0.9,
                 num_iterations_per_update=10,
                 show_progress_bar=True,
                 use_distortions=True,
                 weight_l2_penalty=0.00005)
    add_argument('syms', help='Symbols table mapping from strings to integers')
    add_argument('tr_img_dir', help='Directory containing word images')
    add_argument('tr_txt_table',
                 help='Character transcriptions of each training image')
    add_argument('va_txt_table',
                 help='Character transcriptions of each validation image')
    args = args()
    laia.logging.config_from_args(args)

    laia.random.manual_seed(args.seed)

    syms = laia.utils.SymbolsTable(args.syms)
    model = build_model(num_outputs=len(syms))
    if args.gpu > 0:
        model = model.cuda(args.gpu - 1)
    else:
        model = model.cpu()

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
            tr_ds, batch_size=1, num_workers=8, shuffle=True,
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
        criterion=None,
        optimizer=optimizer,
        data_loader=tr_ds_loader,
        progress_bar='Train' if args.show_progress_bar else False)

    evaluator = laia.engine.Evaluator(
        model=model,
        data_loader=va_ds_loader,
        progress_bar='Valid' if args.show_progress_bar else False)

    engine_wrapper = laia.engine.HtrEngineWrapper(trainer, evaluator, gpu=args.gpu)

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
    """
    trainer.set_epoch_saver_trigger(
        SaverTriggerCollection(
            SaverTrigger(EveryEpoch(trainer, 10),
                         LastParametersSaver('./checkpoint')),
            SaverTrigger(MeterDecrease(engine_wrapper.valid_cer),
                         ParametersSaver('./checkpoint-best-valid-cer')),
            SaverTrigger(MeterDecrease(engine_wrapper.train_cer),
                         ParametersSaver('./checkpoint-best-train-cer'))))
    """

    # Start training
    engine_wrapper.run()
