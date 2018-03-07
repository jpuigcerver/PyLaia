#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division

import os

import torch
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence

import laia.data
import laia.engine
import laia.nn
import laia.utils
from laia.data import PaddedTensor
from laia.engine.triggers import Any, EveryEpoch, MaxEpochs, \
    MeterStandardDeviation, MeterDecrease
from laia.savers import SaverTrigger, SaverTriggerCollection
from laia.utils.arguments import add_argument, add_defaults, args


class Model(torch.nn.Module):
    def __init__(self, num_output_symbols, fixed_rows=1):
        super(Model, self).__init__()
        self._conv = torch.nn.Sequential(
            # conv1_1
            torch.nn.Conv2d(1, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            # conv1_2
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
            # conv2_1
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            # conv2_2
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
            # conv3_1
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            # conv3_2
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            # conv3_3
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            # conv3_4
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            # conv3_5
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            # conv3_6
            # torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # torch.nn.ReLU(inplace=True),
            # conv4_1
            torch.nn.Conv2d(256, 512, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            # conv4_2
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            # conv4_3
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True)
        )

        self._collapse = torch.nn.Sequential(
            # Transform colvolved images to have fixed height
            laia.nn.AdaptiveAvgPool2d(output_sizes=(fixed_rows, None)),
            # Transform image into a sequence of columns
            laia.nn.ImageColumnsToSequence(rows=fixed_rows,
                                           return_packed=False))
        # LSTM
        self._lstm = torch.nn.LSTM(
            input_size=fixed_rows * 512, hidden_size=512,
            num_layers=2, dropout=0.5, bidirectional=True)

        # Final linear classifier
        self._linear = torch.nn.Linear(2 * 512, num_output_symbols)

    def forward(self, x):
        x, xs = (x.data, x.sizes) if isinstance(x, PaddedTensor) else (x, None)
        x = self._conv(x)
        xs = (xs / 2) / 2

        x = x.mean(dim=2).permute(2, 0, 1).contiguous()
        xs = xs[:, 1].contiguous().data.tolist()

        # x, xs = self._collapse(PaddedTensor(data=x, sizes=xs))
        # xs is the number of timestep of each sample.

        x = torch.nn.functional.dropout(x, p=0.5,
                                        training=self.training)
        x, _ = self._lstm(pack_padded_sequence(x, xs))

        # xs is now the number of samples in each timestep
        xs = x.batch_sizes

        x = torch.nn.functional.dropout(x.data, p=0.5,
                                        training=self.training)
        x = self._linear(x)
        return PackedSequence(x, xs)


if __name__ == '__main__':
    add_defaults()
    add_argument('syms')
    add_argument('tr_img_dir')
    add_argument('tr_txt_table')
    add_argument('va_txt_table')
    args = args()

    laia.manual_seed(args.seed)

    syms = laia.utils.SymbolsTable(args.syms)

    model = Model(num_output_symbols=len(syms))
    if args.gpu > 0:
        model = model.cuda(args.gpu - 1)
    else:
        model = model.cpu()

    ctc = laia.losses.CTCLoss()

    parameters = model.parameters()
    optimizer = torch.optim.RMSprop(parameters, lr=args.learning_rate,
                                    momentum=args.momentum)

    tr_ds = laia.data.TextImageFromTextTableDataset(
        args.tr_txt_table, args.tr_img_dir,
        img_transform=laia.utils.ImageToTensor(),
        txt_transform=laia.utils.TextToTensor(syms))
    tr_ds_loader = torch.utils.data.DataLoader(
        tr_ds, args.batch_size, num_workers=8,
        collate_fn=laia.data.PaddingCollater({
            'img': [1, None, None]
        }, sort_key=lambda x: -x['img'].size(2)),
        shuffle=True)

    va_ds = laia.data.TextImageFromTextTableDataset(
        args.va_txt_table, args.tr_img_dir,
        img_transform=laia.utils.ImageToTensor(),
        txt_transform=laia.utils.TextToTensor(syms))
    va_ds_loader = torch.utils.data.DataLoader(
        va_ds, args.batch_size, num_workers=8,
        collate_fn=laia.data.PaddingCollater({
            'img': [1, None, None]
        }, sort_key=lambda x: -x['img'].size(2)))


    def batch_input_fn(batch):
        assert (isinstance(batch['img'], laia.data.PaddedTensor) or
                torch.is_tensor(batch['img']))
        if isinstance(batch['img'], laia.data.PaddedTensor):
            x = batch['img'].data
            xs = batch['img'].sizes[:, 1:].contiguous()
            if args.gpu > 0:
                x, xs = x.cuda(args.gpu - 1), xs.cuda(args.gpu - 1)
            else:
                x, xs = x.cpu(), xs.cpu()
            return laia.data.PaddedTensor(data=Variable(x), sizes=Variable(xs))
        else:
            if args.gpu > 0:
                x = batch['img'].cuda(args.gpu - 1)
            else:
                x = batch['img'].cpu()
            return Variable(x)


    def batch_target_fn(batch):
        return batch['txt']


    trainer = laia.engine.Trainer(model=model,
                                  criterion=ctc,
                                  optimizer=optimizer,
                                  data_loader=tr_ds_loader,
                                  batch_input_fn=batch_input_fn,
                                  batch_target_fn=batch_target_fn)

    evaluator = laia.engine.Evaluator(model=model,
                                      data_loader=va_ds_loader,
                                      batch_input_fn=batch_input_fn,
                                      batch_target_fn=batch_target_fn)

    engine_wrapper = laia.engine.HtrEngineWrapper(trainer, evaluator)

    # List of early stop triggers.
    # If any of these returns True, training will stop.
    early_stop_triggers = []

    # Configure MaxEpochs trigger
    if args.max_epochs and args.max_epochs > 0:
        early_stop_triggers.append(
            MaxEpochs(trainer=trainer, max_epochs=args.max_epochs))

    # Configure MeterStandardDeviation trigger to monitor validation CER
    if args.valid_cer_std_values and args.valid_cer_std_threshold:
        early_stop_triggers.append(
            MeterStandardDeviation(
                meter=engine_wrapper.valid_cer,
                threshold=args.valid_cer_std_threshold,
                num_values_to_keep=args.valid_cer_std_values))

    trainer.set_early_stop_trigger(Any(*early_stop_triggers))


    class LastParametersSaver(object):
        def __init__(self, base_path, keep_checkpoints=5):
            self._base_path = base_path
            self._last_checkpoints = []
            self._keep_checkpoints = keep_checkpoints
            self._nckpt = 0

        def __call__(self, trainer):
            path = '{}-{}'.format(self._base_path, trainer.epochs)
            print('Saving model parameters to {!r}'.format(path))
            try:
                torch.save(trainer.model.state_dict(), path)
            except:
                # TODO(jpuigcerver): Log error saving new checkpoint
                return False

            if len(self._last_checkpoints) < self._keep_checkpoints:
                self._last_checkpoints.append(path)
            else:
                print('Removing old parameters remove: {!r}'.format(
                    self._last_checkpoints[self._nckpt]))
                try:
                    os.remove(self._last_checkpoints[self._nckpt])
                except:
                    # TODO(jpuigcerver): Log error deleting old checkpoint
                    pass
                self._last_checkpoints[self._nckpt] = path
                self._nckpt = (self._nckpt + 1) % self._keep_checkpoints

            return True


    class ParametersSaver(object):
        def __init__(self, path):
            self._path = path

        def __call__(self, trainer):
            print('Saving model parameters to {!r}'.format(self._path))
            try:
                torch.save(trainer.model.state_dict(), self._path)
                return True
            except:
                # TODO(jpuigcerver): Log error saving checkpoint
                return False


    trainer.set_epoch_saver_trigger(
        SaverTriggerCollection(
            SaverTrigger(EveryEpoch(trainer, 10),
                         LastParametersSaver('./checkpoint')),
            SaverTrigger(MeterDecrease(engine_wrapper.valid_cer),
                         ParametersSaver('./checkpoint-best-valid-cer')),
            SaverTrigger(MeterDecrease(engine_wrapper.train_cer),
                         ParametersSaver('./checkpoint-best-train-cer'))))

    # Start training
    engine_wrapper.run()
