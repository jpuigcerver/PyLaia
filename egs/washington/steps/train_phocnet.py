#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division

import argparse
import laia.data
import laia.engine
import laia.nn
import laia.utils
import os
import torch
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence
from laia.data import PaddedTensor

from laia.losses.loss import Loss
from laia.engine.triggers import Any, EveryEpoch, MaxEpochs, MeterStandardDeviation, MeterDecrease

from laia.savers import SaverTrigger, SaverTriggerCollection

from laia.meters import AllPairsMetricAveragePrecisionMeter


def Model(phoc_size):
    return torch.nn.Sequential(
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
        torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
        torch.nn.ReLU(inplace=True),
        # conv4_1
        torch.nn.Conv2d(256, 512, kernel_size=3, padding=1),
        torch.nn.ReLU(inplace=True),
        # conv4_2
        torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
        torch.nn.ReLU(inplace=True),
        # conv4_3
        torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
        torch.nn.ReLU(inplace=True),
        # SPP layer
        laia.nn.PyramidMaxPool2d(levels=3),
        # Linear layers
        torch.nn.Linear(512 * (3 + 2 +1), 4096),
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(),
        torch.nn.Linear(4096, 4096),
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(),
        torch.nn.Linear(4096, phoc_size),
        # Predicted PHOC
        torch.nn.Sigmoid()
    )


class TensorArgumentToDevice(object):
    def __init__(self, argname, device, requires_grad=False):
        self._argname = argname
        self._device = device
        self._requires_grad = requires_grad

    def __call__(self, batch):
        out = batch[self._argname]
        if isinstance(out, PaddedTensor):
            x = Variable(out.data, requires_grad=self._requires_grad)
            xs = Variable(out.sizes, requires_grad=self._requires_grad)
            if self._device > 0:
                return PaddedTensor(data=x.cuda(self._device - 1),
                                    sizes=xs.cuda(self._device - 1))
            else:
                return PaddedTensor(data=x.cpu(), sizes=xs.cpu())
        else:
            if self._device > 0:
                return Variable(out.cuda(self._device - 1))
            else:
                return Variable(out.cpu())


class PaddedImageArgumentToDevice(TensorArgumentToDevice):
    def __init__(self, argname, device, requires_grad=False,
                 keep_channels=False):
        super(PaddedImageArgumentToDevice, self).__init__(
            argname, device, requires_grad)
        self._keep_channels = keep_channels

    def __call__(self, batch):
        out = super(PaddedImageArgumentToDevice, self).__call__(batch)
        assert isinstance(out, PaddedTensor) or torch.is_tensor(out)
        if self._keep_channel or not isinstance(out, PaddedTensor):
            return out
        else:
            return PaddedTensor(data=out.data,
                                sizes=out.sizes[:, 1:].contiguous())

class IgnorePaddingArgumentToDevice(TensorArgumentToDevice):
    def __init__(self, argname, device, requires_grad=False):
        super(IgnorePaddingArgumentToDevice, self).__init__(
            argname, device, requires_grad)

    def __call__(self, batch):
        out = super(IgnorePaddingArgumentToDevice, self).__call__(batch)
        if isinstance(out, PaddedTensor):
            return out.data
        else:
            return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0005,
                        help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0,
                        help='Momentum')
    parser.add_argument('--gpu', type=int, default=1,
                        help='Use this GPU (starting from 1)')
    parser.add_argument('--seed', type=int, default=0x12345,
                        help='Seed for random number generators')
    parser.add_argument('--max_epochs', type=int, default=None,
                        help='Maximum number of training epochs')
    parser.add_argument('--phoc_levels', type=int, default=[1,2,3,4,5],
                        nargs='+',
                        help='PHOC levels used to encode the transcript')
    parser.add_argument('syms')
    parser.add_argument('tr_img_dir')
    parser.add_argument('tr_txt_table')
    parser.add_argument('va_txt_table')
    args = parser.parse_args()

    laia.manual_seed(args.seed)

    syms = laia.utils.SymbolsTable(args.syms)

    loss = torch.nn.BCELoss()
    phoc_size = sum(args.phoc_levels) * len(syms)
    model = Model(phoc_size)
    if args.gpu > 0:
        model = model.cuda(args.gpu - 1)
    else:
        model = model.cpu()

    parameters = model.parameters()
    optimizer = torch.optim.RMSprop(parameters, lr=args.learning_rate,
                                    momentum=args.momentum)

    tr_ds = laia.data.TextImageFromTextTableDataset(
        args.tr_txt_table, args.tr_img_dir,
        img_transform=laia.utils.ImageToTensor(),
        txt_transform=laia.utils.TextToPHOC(syms, args.phoc_levels))
    tr_ds_loader = torch.utils.data.DataLoader(
        tr_ds, args.batch_size, num_workers=8,
        collate_fn=laia.data.PaddingCollater({
            'img': [1, None, None],
            'txt': [phoc_size],
        }, sort_key=lambda x: -x['img'].size(2)),
        shuffle=True)

    va_ds = laia.data.TextImageFromTextTableDataset(
        args.va_txt_table, args.tr_img_dir,
        img_transform=laia.utils.ImageToTensor(),
        txt_transform=laia.utils.TextToPHOC(syms, args.phoc_levels))
    va_ds_loader = torch.utils.data.DataLoader(
        va_ds, args.batch_size, num_workers=8,
        collate_fn=laia.data.PaddingCollater({
            'img': [1, None, None],
            'txt': [phoc_size],
        }, sort_key=lambda x: -x['img'].size(2)))



    # List of early stop triggers.
    # If any of these returns True, training will stop.
    early_stop_triggers = []

    # Configure MaxEpochs trigger
    if args.max_epochs and args.max_epochs > 0:
        early_stop_triggers.append(
            MaxEpochs(trainer=trainer, max_epochs=args.max_epochs))


    batch_input_fn = IgnorePaddingArgumentToDevice('img', args.gpu)
    batch_target_fn = TensorArgumentToDevice('txt', args.gpu)

    trainer = laia.engine.Trainer(model=model,
                                  criterion=loss,
                                  optimizer=optimizer,
                                  dataset=tr_ds_loader,
                                  batch_input_fn=batch_input_fn,
                                  batch_target_fn=batch_target_fn)

    evaluator = laia.engine.Evaluator(model=model,
                                      dataset=va_ds_loader,
                                      batch_input_fn=batch_input_fn,
                                      batch_target_fn=batch_target_fn)




    trainer.set_early_stop_trigger(Any(*early_stop_triggers))


    """
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
    """
    trainer.run()
